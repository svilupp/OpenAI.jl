# Streaming Tests Todo List

## Basic Functionality
- [x] Test basic function callback streaming
- [x] Test IO streaming to buffer
- [x] Test StreamCallback object streaming
- [ ] Test backward compatibility with existing streamcallback usage

## Toy Streaming Server
- [x] Create mock OpenAI streaming server
- [x] Test server with different response patterns
- [x] Test server with error conditions
- [x] Test server with [DONE] messages

## StreamCallbacks.jl Integration
- [x] Test OpenAIStream flavor parsing
- [x] Test custom chunk processing
- [x] Test response building from chunks
- [ ] Test different sink types

## Error Handling
- [x] Test connection errors during streaming
- [x] Test malformed response handling
- [x] Test early disconnection recovery
- [x] Test rate limit handling

## Response Building
- [x] Test OpenAIResponse construction from streamed chunks
- [x] Test delta merging in responses
- [x] Test role and content field handling
- [x] Test empty response handling

## Edge Cases
- [ ] Test very large responses
- [ ] Test rapid message sequences
- [ ] Test unicode content
- [ ] Test empty messages
