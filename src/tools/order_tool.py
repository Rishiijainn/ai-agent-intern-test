import json
import re
from typing import Dict, Any, Optional
from src.config import ORDERS_FILE

class OrderLookupTool:
    def __init__(self, orders_file_path=ORDERS_FILE):
        self.orders_file_path = orders_file_path
        self._orders_cache = None
    def _load_orders(self) -> Dict[str, Any]:
        """Lazy load orders from data/orders.json"""
        if self._orders_cache is None:
            if not self.orders_file_path.exists():
                print(f"Warning: Orders file not found at {self.orders_file_path}")
                return {}
            try:
                with open(self.orders_file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # Case 1: Agar list of orders ho: [ { "order_id": ... } ]
                    if isinstance(data, list):
                        self._orders_cache = {
                            str(item.get("order_id", "")).strip().upper(): item 
                            for item in data if isinstance(item, dict) and "order_id" in item
                        }
                    # Case 2: Agar dict ho: { "orders": [ ... ] }
                    elif isinstance(data, dict):
                        if "orders" in data and isinstance(data["orders"], list):
                            self._orders_cache = {
                                str(item.get("order_id", "")).strip().upper(): item 
                                for item in data["orders"] if isinstance(item, dict) and "order_id" in item
                            }
                        else:
                            self._orders_cache = {
                                str(k).strip().upper(): v for k, v in data.items()
                            }
                    else:
                        self._orders_cache = {}
            except Exception as e:
                print(f"Error loading orders: {e}")
                self._orders_cache = {}
        return self._orders_cache
    def normalize_order_id(self, order_id: Optional[str]) -> Optional[str]:
        """Strip whitespace, convert to uppercase, handle variations like ORD 1007 or ord-1007"""
        if not order_id or not isinstance(order_id, str):
            return None
        cleaned = order_id.strip().upper()
        # Normalize spaces or missing hyphens if needed (e.g., ORD 1007 -> ORD-1007)
        match = re.search(r"ORD[-_\s]?(\d+)", cleaned)
        if match:
            return f"ORD-{match.group(1)}"
        return cleaned

    def lookup(self, order_id: Optional[str]) -> Dict[str, Any]:
        """
        Executes sanitized order lookup.
        Strict Privacy: Drops customer_email, address, internal_notes, risk_score.
        Authoritative Status: Removes stale delivery estimates for cancelled/returned orders.
        """
        if not order_id or str(order_id).strip() == "":
            return {
                "success": False,
                "error": "missing_order_id",
                "message": "Order ID was not provided. Please ask the customer for their order ID."
            }

        norm_id = self.normalize_order_id(order_id)
        orders = self._load_orders()

        if not norm_id or norm_id not in orders:
            return {
                "success": False,
                "error": "order_not_found",
                "message": f"No order found with ID '{order_id}'. Please check the ID and try again."
            }

        raw_order = orders[norm_id]
        status = str(raw_order.get("status", "Unknown")).upper()

        # Parse items list safely
        items = raw_order.get("items", [])
        items_summary = []
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    name = it.get("name") or it.get("product_name") or it.get("title", "Item")
                    qty = it.get("quantity", 1)
                    items_summary.append(f"{qty}x {name}")
                elif isinstance(it, str):
                    items_summary.append(it)

        # Base sanitized data (WHITELIST ONLY)
        sanitized = {
            "order_id": norm_id,
            "status": status,
            "items_summary": items_summary,
            "order_date": raw_order.get("order_date") or raw_order.get("created_at"),
            "shipping_method": raw_order.get("shipping_method"),
            "carrier": raw_order.get("carrier")
        }

        # Status-aware delivery logic (Do not report stale delivery dates on cancelled/returned orders)
        if status in ["CANCELLED", "CANCELED", "RETURNED"]:
            sanitized["estimated_delivery"] = None
            sanitized["tracking_number"] = None
            sanitized["delivery_notice"] = f"This order is {status.lower()}. No active delivery is scheduled."
        else:
            sanitized["estimated_delivery"] = raw_order.get("estimated_delivery") or raw_order.get("delivery_date")
            sanitized["tracking_number"] = raw_order.get("tracking_number")

        return {
            "success": True,
            "data": sanitized
        }


# Tool definition for OpenAI Function Calling
ORDER_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "lookup_order",
        "description": "Look up current status and delivery details for a customer's order using an Order ID (e.g. ORD-1007). Do not guess order details without calling this tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The customer's order ID (e.g., ORD-1007, ord-1002)."
                }
            },
            "required": ["order_id"]
        }
    }
}

if __name__ == "__main__":
    tool = OrderLookupTool()
    print("Total Orders Loaded:", len(tool._load_orders()))
    print("Testing Normal Lookup:", tool.lookup("ORD-1001"))