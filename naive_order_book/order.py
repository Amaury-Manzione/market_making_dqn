class Order:
    def __init__(self, order_type, price, quantity, agent_id=None):
        """Construct an object of class Order

        Args:
            order_type (_type_): buy order : 1 or sell order : -1
            price (_type_): price of the order
            quantity (_type_): units of asset of the order
            agent_id (_type_, optional): id of the agent None for noisy agents and str for market makers
            Defaults to None.
        """
        self.order_type = order_type
        self.price = price
        self.quantity = quantity
        self.agent_id = agent_id
