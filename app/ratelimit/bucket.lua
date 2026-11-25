-- Atomic Token Bucket Rate Limiter (Day 35)
-- KEYS[1]: Redis hash key (e.g., "ratelimit:<key_id>")
-- ARGV[1]: capacity (max burst tokens)
-- ARGV[2]: refill_rate (tokens per second)
-- ARGV[3]: cost (tokens per request, default 1.0)
-- ARGV[4]: optional timestamp override (empty string -> use Redis server TIME)

local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3]) or 1.0
local now_override = ARGV[4]

-- Use consistent server-side Redis TIME unless an explicit override is supplied
local now
if now_override ~= nil and now_override ~= "" then
    now = tonumber(now_override)
else
    local t = redis.call("TIME")
    now = tonumber(t[1]) + (tonumber(t[2]) / 1000000.0)
end

-- Read current bucket state from Redis Hash
local data = redis.call("HMGET", key, "tokens", "last_refill")
local tokens = tonumber(data[1])
local last_refill = tonumber(data[2])

-- Initialize new bucket as full
if tokens == nil or last_refill == nil then
    tokens = capacity
    last_refill = now
end

-- Lazy refill capped at capacity
local elapsed = math.max(0.0, now - last_refill)
tokens = math.min(capacity, tokens + (elapsed * refill_rate))
last_refill = now

local allowed = 0
local retry_after = "0"

if tokens >= cost then
    tokens = tokens - cost
    allowed = 1
else
    local deficit = cost - tokens
    retry_after = tostring(deficit / refill_rate)
end

-- Persist updated state atomically in Redis Hash + set idle TTL
redis.call("HSET", key, "tokens", tostring(tokens), "last_refill", tostring(last_refill))
local ttl = math.ceil((capacity / refill_rate) * 2)
if ttl < 60 then
    ttl = 60
end
redis.call("EXPIRE", key, ttl)

-- Return strings for floats so Redis Lua does not truncate decimals
return { allowed, retry_after, tostring(math.floor(tokens)) }
