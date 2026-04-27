class Family:
    def __init__(self, members=None):
        """
        Initializes a family with optional members.

        Args:
            members (list): List of Agent instances
        """
        self.members = set()

        if members:
            for a in members:
                self.add_member(a)  # ensures proper linking

    def add_member(self, a):
        """
        Adds an agent to the family and links the family to the agent.
        """
        self.members.add(a)
        a.family = self

    def remove_member(self, a):
        """
        Removes an agent from the family and breaks the link on the agent side.
        """
        self.members.discard(a)
        a.family = None

    def __len__(self):
        return len(self.members)

    def __iter__(self):
        return iter(self.members)

    def __repr__(self):
        return f"Family({[a.name for a in self.members]})"
