using Test
using OpenAI
using HTTP
using JSON3
using StreamCallbacks

# Toy Streaming Server
function create_streaming_server(responses; error_after=nothing, malformed_response=false)
    router = HTTP.Router()
    HTTP.register!(router, "POST", "/v1/chat/completions", function(req)
        headers = [
            "Content-Type" => "text/event-stream",
            "Connection" => "keep-alive",
            "Cache-Control" => "no-cache"
        ]

        # Simulate rate limit error
        if haskey(req.headers, "X-Test-Rate-Limit")
            return HTTP.Response(429, "Too Many Requests")
        end

        return HTTP.Response(200, headers; body=IOBuffer(), response_stream=HTTP.Stream(function(stream)
            for (i, response) in enumerate(responses)
                # Simulate connection error
                if !isnothing(error_after) && i > error_after
                    close(stream.io)
                    return
                end

                # Simulate streaming delay
                sleep(0.1)

                # Send malformed response if requested
                if malformed_response
                    write(stream, "data: {invalid_json}\n\n")
                    continue
                end

                # Format as SSE
                chunk = Dict("choices" => [Dict("delta" => response)])
                write(stream, "data: $(JSON3.write(chunk))\n\n")
            end
            write(stream, "data: [DONE]\n\n")
            close(stream.io)
        end))
    end)

    server = HTTP.serve!(router, "127.0.0.1", 8444)
    return server
end

@testset "Streaming" begin
    # Test basic function callback streaming
    @testset "Basic Function Callback" begin
        # Create a mock server with predefined responses
        responses = [
            Dict("role" => "assistant"),
            Dict("content" => "Hello"),
            Dict("content" => " World"),
            Dict("content" => "!")
        ]

        server = create_streaming_server(responses)
        try
            # Collect streamed chunks
            chunks = String[]
            provider = OpenAIProvider(
                api_key="test",
                base_url="http://127.0.0.1:8444/v1"
            )

            # Test with basic function callback
            response = create_chat(
                provider,
                "gpt-4",
                [Dict("role" => "user", "content" => "Say hello")];
                streamcallback = chunk -> push!(chunks, chunk)
            )

            # Verify response structure
            @test response.status == 200
            @test length(response.response) == length(responses)

            # Verify chunks were received correctly
            @test length(chunks) == length(responses)
            @test any(contains.(chunks, "Hello"))
            @test any(contains.(chunks, "World"))
            @test any(contains.(chunks, "!"))
        finally
            close(server)
        end
    end

    # Test IO streaming
    @testset "IO Streaming" begin
        responses = [
            Dict("role" => "assistant"),
            Dict("content" => "Test"),
            Dict("content" => " Message")
        ]

        server = create_streaming_server(responses)
        try
            # Test streaming to IOBuffer
            output = IOBuffer()
            provider = OpenAIProvider(
                api_key="test",
                base_url="http://127.0.0.1:8444/v1"
            )

            response = create_chat(
                provider,
                "gpt-4",
                [Dict("role" => "user", "content" => "Test")];
                streamcallback = output
            )

            # Verify response
            @test response.status == 200

            # Verify IO contents
            content = String(take!(output))
            @test contains(content, "Test Message")
        finally
            close(server)
        end
    end

    # Test StreamCallbacks.jl integration
    @testset "StreamCallbacks Integration" begin
        responses = [
            Dict("role" => "assistant"),
            Dict("content" => "Stream"),
            Dict("content" => "Callbacks"),
            Dict("content" => "Test")
        ]

        server = create_streaming_server(responses)
        try
            chunks = []
            callback = StreamCallback(
                out = chunk -> push!(chunks, chunk),
                flavor = OpenAIStream()
            )

            provider = OpenAIProvider(
                api_key="test",
                base_url="http://127.0.0.1:8444/v1"
            )

            response = create_chat(
                provider,
                "gpt-4",
                [Dict("role" => "user", "content" => "Test StreamCallbacks")];
                streamcallback = callback
            )

            # Verify response
            @test response.status == 200
            @test length(chunks) == length(responses)

            # Verify chunk processing
            @test any(chunk -> chunk.content == "Stream", chunks)
            @test any(chunk -> chunk.content == "Callbacks", chunks)
            @test any(chunk -> chunk.content == "Test", chunks)
        finally
            close(server)
        end
    end

    # Test error handling
    @testset "Error Handling" begin
        # Test connection errors
        @testset "Connection Errors" begin
            responses = [
                Dict("role" => "assistant"),
                Dict("content" => "This"),
                Dict("content" => " will"),
                Dict("content" => " fail")
            ]

            server = create_streaming_server(responses, error_after=2)
            try
                provider = OpenAIProvider(
                    api_key="test",
                    base_url="http://127.0.0.1:8444/v1"
                )

                chunks = String[]
                @test_throws HTTP.IOError create_chat(
                    provider,
                    "gpt-4",
                    [Dict("role" => "user", "content" => "Test connection error")];
                    streamcallback = chunk -> push!(chunks, chunk)
                )

                # Verify we received some chunks before error
                @test length(chunks) > 0
                @test length(chunks) <= 3
            finally
                close(server)
            end
        end

        # Test malformed response handling
        @testset "Malformed Response" begin
            responses = [Dict("content" => "test")]
            server = create_streaming_server(responses, malformed_response=true)
            try
                provider = OpenAIProvider(
                    api_key="test",
                    base_url="http://127.0.0.1:8444/v1"
                )

                chunks = String[]
                @test_throws ArgumentError create_chat(
                    provider,
                    "gpt-4",
                    [Dict("role" => "user", "content" => "Test malformed")];
                    streamcallback = chunk -> push!(chunks, chunk)
                )
            finally
                close(server)
            end
        end

        # Test rate limit handling
        @testset "Rate Limit" begin
            server = create_streaming_server([Dict("content" => "test")])
            try
                provider = OpenAIProvider(
                    api_key="test",
                    base_url="http://127.0.0.1:8444/v1"
                )

                @test_throws HTTP.StatusError create_chat(
                    provider,
                    "gpt-4",
                    [Dict("role" => "user", "content" => "Test rate limit")];
                    streamcallback = chunk -> nothing,
                    additional_headers=["X-Test-Rate-Limit" => "true"]
                )
            finally
                close(server)
            end
        end
    end

    # Test response building
    @testset "Response Building" begin
        # Test OpenAIResponse construction and delta merging
        @testset "Response Construction" begin
            responses = [
                Dict("role" => "assistant"),
                Dict("content" => "Hello"),
                Dict("content" => " World"),
                Dict("content" => "!")
            ]

            server = create_streaming_server(responses)
            try
                chunks = []
                provider = OpenAIProvider(
                    api_key="test",
                    base_url="http://127.0.0.1:8444/v1"
                )

                response = create_chat(
                    provider,
                    "gpt-4",
                    [Dict("role" => "user", "content" => "Test response")];
                    streamcallback = chunk -> push!(chunks, chunk)
                )

                # Test response structure
                @test response.status == 200
                @test length(response.response) == length(responses)

                # Test delta merging
                deltas = map(r -> r["choices"][1]["delta"], response.response)
                @test deltas[1]["role"] == "assistant"
                @test haskey(deltas[2], "content") && deltas[2]["content"] == "Hello"
                @test haskey(deltas[3], "content") && deltas[3]["content"] == " World"
                @test haskey(deltas[4], "content") && deltas[4]["content"] == "!"
            finally
                close(server)
            end
        end

        # Test role and content field handling
        @testset "Field Handling" begin
            responses = [
                Dict("role" => "assistant"),
                Dict("content" => "Test"),
                Dict(),  # Empty delta
                Dict("content" => " Content")
            ]

            server = create_streaming_server(responses)
            try
                provider = OpenAIProvider(
                    api_key="test",
                    base_url="http://127.0.0.1:8444/v1"
                )

                response = create_chat(
                    provider,
                    "gpt-4",
                    [Dict("role" => "user", "content" => "Test fields")];
                    streamcallback = chunk -> nothing
                )

                # Test field handling
                deltas = map(r -> r["choices"][1]["delta"], response.response)
                @test haskey(deltas[1], "role")
                @test !haskey(deltas[1], "content")
                @test haskey(deltas[2], "content")
                @test !haskey(deltas[2], "role")
                @test isempty(deltas[3])
                @test haskey(deltas[4], "content")
            finally
                close(server)
            end
        end

        # Test empty response handling
        @testset "Empty Response" begin
            responses = [Dict("role" => "assistant")]  # Only role, no content

            server = create_streaming_server(responses)
            try
                provider = OpenAIProvider(
                    api_key="test",
                    base_url="http://127.0.0.1:8444/v1"
                )

                response = create_chat(
                    provider,
                    "gpt-4",
                    [Dict("role" => "user", "content" => "Test empty")];
                    streamcallback = chunk -> nothing
                )

                # Test empty response handling
                @test length(response.response) == 1
                delta = response.response[1]["choices"][1]["delta"]
                @test haskey(delta, "role")
                @test !haskey(delta, "content")
            finally
                close(server)
            end
        end
    end
end
