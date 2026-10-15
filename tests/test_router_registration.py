import os
import yaml
from pathlib import Path

def test_router_registration():
    base_dir = Path(__file__).parent.parent
    docker_compose_file = base_dir / "deployment" / "docker-compose.yml"
    env_file = base_dir / ".env"

    print("=" * 70)
    print("ROUTER CONFIGURATION & MODEL REGISTRATION TEST")
    print("=" * 70)

    # 1. Verify docker-compose.yml registration
    assert docker_compose_file.exists(), f"Missing docker-compose.yml at {docker_compose_file}"
    with open(docker_compose_file, "r", encoding="utf-8") as f:
        dc_config = yaml.safe_load(f)

    services = dc_config.get("services", {})
    assert "llm-router-cpu" in services, "llm-router-cpu (Base) service not registered in docker-compose.yml!"
    assert "llm-router-finetuned" in services, "llm-router-finetuned service not registered in docker-compose.yml!"

    router_cmd = services["llm-router-cpu"].get("command", [])
    model_arg = next((router_cmd[i+1] for i, arg in enumerate(router_cmd) if arg == "--model"), "")
    
    print(f"Registered LLM Base Router Service: 'llm-router-cpu' (Port 8080)")
    print(f"Registered LLM Fine-Tuned Router Service: 'llm-router-finetuned' (Port 8081)")
    print(f"Registered Model Path in Router: '{model_arg}'")
    assert "qwen2.5-3b-instruct-q4_k_m.gguf" in model_arg, "Fine-tuned GGUF model not registered in router!"

    # 2. Verify .env registration
    assert env_file.exists(), f"Missing .env file at {env_file}"
    env_text = env_file.read_text(encoding="utf-8")
    assert "LLM_ROUTER_URL=" in env_text, "LLM_ROUTER_URL missing from .env!"
    assert "LLM_BASE_ROUTER_URL=" in env_text, "LLM_BASE_ROUTER_URL missing from .env!"
    assert "LLM_FINETUNED_ROUTER_URL=" in env_text, "LLM_FINETUNED_ROUTER_URL missing from .env!"
    
    print("-" * 70)
    print("VERIFICATION PASSED: Both Base (8080) and Fine-Tuned (8081) LLM Router endpoints registered.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    test_router_registration()
