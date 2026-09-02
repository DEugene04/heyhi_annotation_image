from datetime import datetime, timezone

from evaluator.openai_pricing import AI_PRICING


def add_project_info(payload, project_id, tokens_used, est_cost, models, service_group_id = 8):
    if type(payload) == list:
        payload = {"payload": payload}
        payload["is_list"] = True
    elif type(payload) == dict:
        payload["is_list"] = False
    payload['project_id'] = project_id
    payload['timestamp'] = datetime.now(timezone.utc).isoformat()
    payload['tokens_used'] = tokens_used
    payload['est_cost'] = est_cost
    payload['model'] = models
    payload['service_group_id'] = service_group_id
    return payload

def calculate_openai_usage_cost(
    model_name: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> float:
    
    try:
        base_model = model_name.lower()
        pricing = AI_PRICING.get(base_model, None)

        if not pricing:
            return 0.0

        cost = (
            (input_tokens * pricing["input"]) + 
            (output_tokens * pricing["output"])
        ) / 1e6

        return cost

    except Exception as e:
        print(f"Error calculating cost in function 'utils.calculate_openai_usage_cost': {e}")
        return 0.0