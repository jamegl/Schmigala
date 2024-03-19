class Edge:
    def __init__(self, first_node, second_node, weight):
        self.first_node = first_node
        self.second_node = second_node
        self.weight = weight

    def get_first_node(self):
        return self.first_node

    def get_second_node(self):
        return self.second_node

    def get_weight(self):
        return self.weight

    def has_node(self, node):
        return (self.first_node == node or self.second_node == node) and self.weight != 0

    def __str__(self):
        return self.first_node.name + self.second_node.name + " - " + str(self.weight)

    def equals(self, other):
        return (self.first_node == other.first_node and self.second_node == other.second_node) \
            or (self.first_node == other.second_node and self.second_node == other.first_node)
