from manim import *
import numpy as np

class XGBoostScene(Scene):
    def construct(self):
        # Title
        title = Text("XGBoost", font_size=36, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        # s
        
        # Subtitle
        subtitle = Text("Extreme Gradient Boosting", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN)
        self.play(Write(subtitle))
        self.wait(0.5)
        
        # Sequential tree building
        trees = VGroup()
        positions = [LEFT * 4, LEFT * 1.5, RIGHT * 1, RIGHT * 3.5]
        colors = [BLUE, GREEN, YELLOW, ORANGE]
        
        # Initial prediction
        pred_tracker = ValueTracker(0)
        prediction_text = always_redraw(lambda: Text(
            f"Prediction: {pred_tracker.get_value():.2f}",
            font_size=24,
            color=YELLOW
        ).to_edge(DOWN))
        
        self.play(Write(prediction_text))
        
        for i, (pos, color) in enumerate(zip(positions, colors)):
            # Tree
            tree = self.create_boosted_tree(color, i + 1)
            tree.shift(pos + UP * 0.5)
            
            # Plus sign between trees
            if i > 0:
                plus = Text("+", font_size=40, color=WHITE)
                plus.move_to((positions[i-1] + positions[i]) / 2 + UP * 0.5)
                self.play(Write(plus))
            
            self.play(FadeIn(tree))
            
            # Update prediction
            new_pred = pred_tracker.get_value() + (0.25 * (i + 1))
            self.play(pred_tracker.animate.set_value(new_pred), run_time=0.8)
            self.wait(0.3)
            
            trees.add(tree)
        
        self.wait(0.5)
        
        # Error reduction visualization
        error_text = Text("Each tree corrects previous errors", font_size=20, color=GREEN)
        error_text.next_to(prediction_text, UP, buff=0.5)
        self.play(Write(error_text))
        
        # Show residuals concept
        residual_arrow = Arrow(
            trees[0].get_bottom() + DOWN * 0.2,
            trees[1].get_top() + UP * 0.2,
            color=RED,
            stroke_width=3
        )
        residual_label = Text("Residuals", font_size=18, color=RED)
        residual_label.next_to(residual_arrow, DOWN, buff=0.1)
        
        self.play(Create(residual_arrow), Write(residual_label))
        
        self.wait(2)
    
    def create_boosted_tree(self, color, number):
        """Create a simple tree with label"""
        # Root
        root = RoundedRectangle(
            height=0.6,
            width=1.2,
            corner_radius=0.1,
            color=color,
            fill_opacity=0.7,
            stroke_width=2
        )
        
        label = Text(f"Tree {number}", font_size=18, color=WHITE)
        label.move_to(root.get_center())
        
        # Simple children representation
        left = Circle(radius=0.15, color=color, fill_opacity=0.5)
        left.shift(DOWN * 0.6 + LEFT * 0.3)
        
        right = Circle(radius=0.15, color=color, fill_opacity=0.5)
        right.shift(DOWN * 0.6 + RIGHT * 0.3)
        
        left_line = Line(root.get_bottom(), left.get_top(), color=color)
        right_line = Line(root.get_bottom(), right.get_top(), color=color)
        
        tree = VGroup(root, label, left, right, left_line, right_line)
        return tree
