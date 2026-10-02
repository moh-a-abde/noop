import XCTest
@testable import Strand

/// Pins the OpenAI-compatible request shape that both the streamed and non-streamed Coach paths
/// share. GPT-5 and later reject `max_tokens` with a 400 and need `max_completion_tokens`; the
/// streamed path used to send the classic shape with no retry, so every interactive Coach message
/// on a GPT-5 model failed. Twin of Android `AiCoachOpenAiParamsTest`; the key lists and the
/// curated model list must match it exactly.
final class AICoachOpenAIParamsTests: XCTestCase {

    private let messages: [[String: Any]] = [["role": "user", "content": "hi"]]

    func testClassicStreamBodySendsTemperatureAndMaxTokens() {
        let body = openAICompatibleChatBody(model: "gpt-4o", messages: messages, modernParams: false, stream: true)
        XCTAssertEqual(body.keys.sorted(), ["max_tokens", "messages", "model", "stream", "temperature"])
        XCTAssertEqual(body["max_tokens"] as? Int, 4096)
        XCTAssertEqual(body["temperature"] as? Double, 0.6)
        XCTAssertEqual(body["stream"] as? Bool, true)
    }

    func testModernStreamBodySendsOnlyMaxCompletionTokens() {
        let body = openAICompatibleChatBody(model: "gpt-5", messages: messages, modernParams: true, stream: true)
        XCTAssertEqual(body.keys.sorted(), ["max_completion_tokens", "messages", "model", "stream"])
        XCTAssertEqual(body["max_completion_tokens"] as? Int, 4096)
        XCTAssertEqual(body["model"] as? String, "gpt-5")
    }

    func testNonStreamBodiesOmitStream() {
        let classic = openAICompatibleChatBody(model: "gpt-4o", messages: messages, modernParams: false, stream: false)
        let modern = openAICompatibleChatBody(model: "gpt-5", messages: messages, modernParams: true, stream: false)
        XCTAssertEqual(classic.keys.sorted(), ["max_tokens", "messages", "model", "temperature"])
        XCTAssertEqual(modern.keys.sorted(), ["max_completion_tokens", "messages", "model"])
    }

    func testReportedGPT5ErrorTriggersModernRetry() {
        XCTAssertTrue(shouldRetryOpenAIModernParams(
            "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."))
        XCTAssertTrue(shouldRetryOpenAIModernParams("Unsupported value: 'temperature'"))
    }

    func testUnrelatedBadRequestDoesNotRetry() {
        XCTAssertFalse(shouldRetryOpenAIModernParams("Invalid model id"))
        XCTAssertFalse(shouldRetryOpenAIModernParams(""))
    }

    func testOpenAICuratedModelsMatchAndroid() {
        XCTAssertEqual(AIProvider.openAI.modelOptions, [
            "gpt-6-astra", "gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna",
            "gpt-5", "gpt-5-mini", "gpt-5-nano",
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            "gpt-4o", "gpt-4o-mini", "o3", "o4-mini"
        ])
        XCTAssertEqual(AIProvider.openAI.defaultModel, "gpt-5-mini")
    }
}
