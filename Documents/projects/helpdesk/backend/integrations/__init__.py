import importlib

def optional_import(name):
    try:
        return importlib.import_module(f"backend.integrations.{name}")
    except ModuleNotFoundError:
        print(f"Optional integration '{name}' not found — skipping.")
        return None

# Load integrations dynamically (only those that exist)
zendesk = optional_import("zendesk")
servicenow = optional_import("servicenow")
freshdesk = optional_import("freshdesk")

__all__ = ["zendesk", "servicenow", "freshdesk"]