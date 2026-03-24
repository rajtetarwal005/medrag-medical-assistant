import redis
import json

# Connect to Redis
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)


def get_chat_history(session_id: str):
    data = redis_client.get(session_id)
    if data:
        return json.loads(data)
    return []


def save_chat_history(session_id: str, history):
    redis_client.set(session_id, json.dumps(history))