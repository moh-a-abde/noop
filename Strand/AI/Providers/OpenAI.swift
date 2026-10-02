import Foundation
import StrandAnalytics

struct OpenAIClient: AIProviderClient {

    func send(
        key: String,
        model: String,
        systemPrompt: String,
        messages: [(role: ChatMessage.Role, content: String)],
        session: URLSession
    ) async throws -> String {
        var wire: [[String: Any]] = [["role": "system", "content": systemPrompt]]
        for m in messages { wire.append(["role": m.role.rawValue, "content": m.content]) }

        // Standard params first (gpt-4 family). Newer/reasoning models reject `temperature` and want
        // `max_completion_tokens`; if the provider 400s about either, retry with the modern shape.
        do {
            return try await chat(key: key, model: model, wire: wire, modernParams: false, session: session)
        } catch let AICoachError.server(code, detail) where code == 400 && shouldRetryOpenAIModernParams(detail) {
            return try await chat(key: key, model: model, wire: wire, modernParams: true, session: session)
        }
    }

    /// K1: Stream via `stream: true`. Same body as `send`, with `stream: true` added, and the same
    /// modern-params retry on a 400 (GPT-5 and later reject `max_tokens`). A 400 arrives before any
    /// SSE line, so no delta has been emitted when the retry starts. SSE parsing via
    /// `SseDeltas.openAiDelta`. Byte-parity pin in `SseDeltasTests.openAiReassembleMatchesFullReply`.
    func stream(
        key: String,
        model: String,
        systemPrompt: String,
        messages: [(role: ChatMessage.Role, content: String)],
        session: URLSession,
        onDelta: (String) -> Void
    ) async throws {
        var wire: [[String: Any]] = [["role": "system", "content": systemPrompt]]
        for m in messages { wire.append(["role": m.role.rawValue, "content": m.content]) }

        do {
            try await streamChat(key: key, model: model, wire: wire, modernParams: false,
                                 session: session, onDelta: onDelta)
        } catch let AICoachError.server(code, detail) where code == 400 && shouldRetryOpenAIModernParams(detail) {
            try await streamChat(key: key, model: model, wire: wire, modernParams: true,
                                 session: session, onDelta: onDelta)
        }
    }

    func fetchModels(key: String, session: URLSession) async throws -> [String] {
        var req = URLRequest(url: AIProvider.openAI.modelsEndpoint)
        req.httpMethod = "GET"
        req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")

        return parseModels(try await performRequest(req, session: session))
    }

    /// Pure: unwrap the `/models` body into chat-capable ids (gpt*/o*). No network — unit-tested.
    func parseModels(_ json: [String: Any]) -> [String] {
        guard let list = json["data"] as? [[String: Any]] else { return [] }
        return list.compactMap { row in
            guard let id = row["id"] as? String, !id.isEmpty else { return nil }
            return (id.hasPrefix("gpt") || id.hasPrefix("o")) ? id : nil
        }
    }

    // MARK: Private

    private func request(key: String, body: [String: Any]) throws -> URLRequest {
        var req = URLRequest(url: AIProvider.openAI.endpoint)
        req.httpMethod = "POST"
        req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        return req
    }

    private func streamChat(
        key: String,
        model: String,
        wire: [[String: Any]],
        modernParams: Bool,
        session: URLSession,
        onDelta: (String) -> Void
    ) async throws {
        let body = openAICompatibleChatBody(model: model, messages: wire, modernParams: modernParams, stream: true)
        try await performStreamingRequest(try request(key: key, body: body), session: session) { payload in
            if let delta = SseDeltas.openAiDelta(payload) {
                onDelta(delta)
            }
        }
    }

    /// `modernParams`: use `max_completion_tokens`, drop `temperature` — required by reasoning models.
    private func chat(
        key: String,
        model: String,
        wire: [[String: Any]],
        modernParams: Bool,
        session: URLSession
    ) async throws -> String {
        // #1074: 900 truncated detailed coaching replies mid-sentence; 4096 lets a full multi-section
        // reply complete (a cap, not a target — the system prompt keeps it short). Matches Gemini + Android.
        let body = openAICompatibleChatBody(model: model, messages: wire, modernParams: modernParams, stream: false)
        let json = try await performRequest(try request(key: key, body: body), session: session)
        guard let choices = json["choices"] as? [[String: Any]],
              let first = choices.first,
              let message = first["message"] as? [String: Any],
              let content = (message["content"] as? String)?
                  .trimmingCharacters(in: .whitespacesAndNewlines), !content.isEmpty else {
            throw emptyReplyError(json)   // #1074: surface the provider's real error if the 200 body has one
        }
        return content
    }
}
