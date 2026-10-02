package com.noop.ai

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the OpenAI-compatible request shape that both the streamed and non-streamed Coach paths
 * share. GPT-5 and later reject `max_tokens` with a 400 and need `max_completion_tokens`; the
 * streamed path used to send the classic shape with no retry, so every interactive Coach message
 * on a GPT-5 model failed. Twin of Swift `AICoachOpenAIParamsTests`; the key lists and the
 * curated model list must match it exactly.
 */
class AiCoachOpenAiParamsTest {

    private val messages = JSONArray().put(JSONObject().put("role", "user").put("content", "hi"))

    private fun sortedKeys(body: JSONObject): List<String> = body.keys().asSequence().toList().sorted()

    @Test
    fun classicStreamBodySendsTemperatureAndMaxTokens() {
        val body = AiCoach.openAiCompatibleChatBody("gpt-4o", messages, modernParams = false, stream = true)
        assertEquals(listOf("max_tokens", "messages", "model", "stream", "temperature"), sortedKeys(body))
        assertEquals(4096, body.getInt("max_tokens"))
        assertEquals(0.6, body.getDouble("temperature"), 0.0)
        assertTrue(body.getBoolean("stream"))
    }

    @Test
    fun modernStreamBodySendsOnlyMaxCompletionTokens() {
        val body = AiCoach.openAiCompatibleChatBody("gpt-5", messages, modernParams = true, stream = true)
        assertEquals(listOf("max_completion_tokens", "messages", "model", "stream"), sortedKeys(body))
        assertEquals(4096, body.getInt("max_completion_tokens"))
        assertEquals("gpt-5", body.getString("model"))
    }

    @Test
    fun nonStreamBodiesOmitStream() {
        val classic = AiCoach.openAiCompatibleChatBody("gpt-4o", messages, modernParams = false, stream = false)
        val modern = AiCoach.openAiCompatibleChatBody("gpt-5", messages, modernParams = true, stream = false)
        assertEquals(listOf("max_tokens", "messages", "model", "temperature"), sortedKeys(classic))
        assertEquals(listOf("max_completion_tokens", "messages", "model"), sortedKeys(modern))
    }

    @Test
    fun reportedGpt5ErrorTriggersModernRetry() {
        val body = """{"error":{"message":"Unsupported parameter: 'max_tokens' is not supported with this model. """ +
            """Use 'max_completion_tokens' instead.","type":"invalid_request_error","param":"max_tokens",""" +
            """"code":"unsupported_parameter"}}"""
        assertTrue(AiCoach.shouldRetryOpenAiModernParams(body))
        assertTrue(AiCoach.shouldRetryOpenAiModernParams("Unsupported value: 'temperature'"))
    }

    @Test
    fun unrelatedBadRequestDoesNotRetry() {
        assertFalse(AiCoach.shouldRetryOpenAiModernParams("""{"error":{"message":"Invalid model id"}}"""))
        assertFalse(AiCoach.shouldRetryOpenAiModernParams(""))
    }

    @Test
    fun openAiCuratedModelsMatchSwift() {
        assertEquals(
            listOf(
                "gpt-6-astra", "gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna",
                "gpt-5", "gpt-5-mini", "gpt-5-nano",
                "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
                "gpt-4o", "gpt-4o-mini", "o3", "o4-mini",
            ),
            AiProvider.OPENAI.models,
        )
        assertEquals("gpt-5-mini", AiProvider.OPENAI.defaultModel)
    }
}
