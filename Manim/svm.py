from manim import *
import numpy as np

class SVMScene(Scene):
    def construct(self):
        # Title
        title = Text("Support Vector Machine (SVM)", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        
        # Create axes
        axes = Axes(
            x_range=[-3, 3],
            y_range=[-3, 3],
            x_length=6,
            y_length=6,
            axis_config={"include_tip": True}
        )
        axes.shift(DOWN * 0.5)
        self.play(Create(axes))
        
        # Generate two classes of data points
        np.random.seed(42)
        class1_data = np.random.randn(15, 2) + np.array([-1, -1])
        class2_data = np.random.randn(15, 2) + np.array([1, 1])
        
        # Create dots for class 1 (blue)
        class1_dots = VGroup(*[
            Dot(axes.c2p(x, y), color=BLUE, radius=0.08)
            for x, y in class1_data
        ])
        
        # Create dots for class 2 (red)
        class2_dots = VGroup(*[
            Dot(axes.c2p(x, y), color=RED, radius=0.08)
            for x, y in class2_data
        ])
        
        self.play(
            LaggedStart(*[FadeIn(dot) for dot in class1_dots], lag_ratio=0.05),
            LaggedStart(*[FadeIn(dot) for dot in class2_dots], lag_ratio=0.05)
        )
        self.wait(0.5)
        
        # Decision boundary (hyperplane)
        boundary = axes.plot(lambda x: x, color=YELLOW, stroke_width=4)
        
        # Support vectors (highlighted)
        support_vectors = VGroup(
            Circle(radius=0.15, color=GREEN).move_to(axes.c2p(-0.5, -0.5)),
            Circle(radius=0.15, color=GREEN).move_to(axes.c2p(0.5, 0.5))
        )
        
        # Margin lines
        margin1 = DashedLine(
            axes.c2p(-3, -3 - 0.7), axes.c2p(3, 3 - 0.7),
            color=GREEN, stroke_width=2
        )
        margin2 = DashedLine(
            axes.c2p(-3, -3 + 0.7), axes.c2p(3, 3 + 0.7),
            color=GREEN, stroke_width=2
        )
        
        self.play(Create(margin1), Create(margin2))
        self.play(Create(boundary))
        self.play(Create(support_vectors))
        
        # Label
        label = Text("Maximum Margin Classifier", font_size=24, color=YELLOW)
        label.next_to(axes, DOWN)
        self.play(Write(label))
        
        self.wait(2)
