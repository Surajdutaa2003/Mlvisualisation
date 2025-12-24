from manim import *

class RandomForestScene(Scene):
    def construct(self):
        # Title
        title = Text("Random Forest", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        # s
        # Subtitle
        subtitle = Text("Ensemble of Decision Trees", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN)
        self.play(Write(subtitle))
        self.wait(0.5)
        
        # Create multiple mini trees
        trees = VGroup()
        colors = [BLUE, GREEN, YELLOW, PURPLE, ORANGE]
        
        for i, color in enumerate(colors):
            tree = self.create_simple_tree(color)
            tree.scale(0.4)
            tree.shift(LEFT * 4 + RIGHT * i * 2 + DOWN * 0.5)
            trees.add(tree)
        
        # Animate trees appearing
        self.play(LaggedStart(*[FadeIn(tree) for tree in trees], lag_ratio=0.2))
        self.wait(0.5)
        
        # Show predictions from each tree
        predictions = VGroup()
        pred_labels = ["Yes", "No", "Yes", "Yes", "No"]
        for i, (tree, pred) in enumerate(zip(trees, pred_labels)):
            pred_text = Text(pred, font_size=20, color=YELLOW if pred == "Yes" else RED)
            pred_text.next_to(tree, DOWN, buff=0.3)
            predictions.add(pred_text)
        
        self.play(LaggedStart(*[Write(pred) for pred in predictions], lag_ratio=0.1))
        self.wait(0.5)
        
        # Voting arrow
        arrow = Arrow(
            predictions.get_bottom() + DOWN * 0.3,
            predictions.get_bottom() + DOWN * 1.5,
            color=WHITE,
            stroke_width=6
        )
        self.play(Create(arrow))
        
        # Final prediction (majority vote)
        final_box = RoundedRectangle(
            height=0.8,
            width=3,
            corner_radius=0.2,
            color=GREEN,
            fill_opacity=0.8,
            stroke_width=3
        )
        final_box.next_to(arrow, DOWN, buff=0.2)
        
        final_text = Text("Final: Yes (3/5)", font_size=28, color=WHITE)
        final_text.move_to(final_box.get_center())
        
        final_group = VGroup(final_box, final_text)
        self.play(FadeIn(final_group))
        
        # Info
        info = Text("Majority voting reduces overfitting", font_size=20, color=GRAY)
        info.to_edge(DOWN)
        self.play(Write(info))
        
        self.wait(2)
    
    def create_simple_tree(self, color):
        """Create a simple tree representation"""
        # Root
        root = Circle(radius=0.3, color=color, fill_opacity=0.8, stroke_width=2)
        
        # Children
        left_child = Circle(radius=0.2, color=color, fill_opacity=0.6, stroke_width=2)
        left_child.shift(DOWN * 0.8 + LEFT * 0.5)
        
        right_child = Circle(radius=0.2, color=color, fill_opacity=0.6, stroke_width=2)
        right_child.shift(DOWN * 0.8 + RIGHT * 0.5)
        
        # Edges
        left_edge = Line(root.get_bottom(), left_child.get_top(), color=color)
        right_edge = Line(root.get_bottom(), right_child.get_top(), color=color)
        
        tree = VGroup(root, left_child, right_child, left_edge, right_edge)
        return tree
