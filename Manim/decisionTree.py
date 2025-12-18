from manim import *

class DecisionTreeScene(Scene):
    def construct(self):
        # Title
        title = Text("Decision Tree", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        
        # Root node
        root = self.create_node("Age > 30?", BLUE)
        root.shift(UP * 2)
        self.play(FadeIn(root))
        self.wait(0.5)
        
        # Edges
        left_edge = Line(root.get_bottom(), root.get_bottom() + DOWN * 1.5 + LEFT * 2)
        right_edge = Line(root.get_bottom(), root.get_bottom() + DOWN * 1.5 + RIGHT * 2)
        
        yes_label = Text("Yes", font_size=18, color=GREEN).next_to(right_edge, RIGHT, buff=0.1)
        no_label = Text("No", font_size=18, color=RED).next_to(left_edge, LEFT, buff=0.1)
        
        self.play(Create(left_edge), Create(right_edge))
        self.play(Write(yes_label), Write(no_label))
        
        # Left child node
        left_node = self.create_node("Income > 50K?", BLUE)
        left_node.move_to(root.get_bottom() + DOWN * 1.5 + LEFT * 2)
        self.play(FadeIn(left_node))
        
        # Right child node (leaf)
        right_node = self.create_node("Approved ✓", GREEN)
        right_node.move_to(root.get_bottom() + DOWN * 1.5 + RIGHT * 2)
        self.play(FadeIn(right_node))
        self.wait(0.5)
        
        # Left child edges
        left_left_edge = Line(left_node.get_bottom(), left_node.get_bottom() + DOWN * 1.5 + LEFT * 1)
        left_right_edge = Line(left_node.get_bottom(), left_node.get_bottom() + DOWN * 1.5 + RIGHT * 1)
        
        yes_label2 = Text("Yes", font_size=16, color=GREEN).next_to(left_right_edge, RIGHT, buff=0.05)
        no_label2 = Text("No", font_size=16, color=RED).next_to(left_left_edge, LEFT, buff=0.05)
        
        self.play(Create(left_left_edge), Create(left_right_edge))
        self.play(Write(yes_label2), Write(no_label2))
        
        # Leaf nodes
        leaf_reject = self.create_node("Rejected ✗", RED)
        leaf_reject.move_to(left_node.get_bottom() + DOWN * 1.5 + LEFT * 1)
        
        leaf_approve = self.create_node("Approved ✓", GREEN)
        leaf_approve.move_to(left_node.get_bottom() + DOWN * 1.5 + RIGHT * 1)
        
        self.play(FadeIn(leaf_reject), FadeIn(leaf_approve))
        
        # Info text
        info = Text("Splits data based on features", font_size=20)
        info.to_edge(DOWN)
        self.play(Write(info))
        
        self.wait(2)
    
    def create_node(self, text, color):
        """Helper function to create a tree node"""
        label = Text(text, font_size=20, color=WHITE)
        box = RoundedRectangle(
            height=label.height + 0.4,
            width=label.width + 0.6,
            corner_radius=0.1,
            color=color,
            fill_opacity=0.8,
            stroke_width=2
        )
        node = VGroup(box, label)
        return node
