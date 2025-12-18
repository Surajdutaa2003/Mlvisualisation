from manim import *
import numpy as np

class KNNScene(Scene):
    def construct(self):
        # Title
        title = Text("K-Nearest Neighbors (KNN)", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        
        # Create axes
        axes = Axes(
            x_range=[0, 10],
            y_range=[0, 10],
            x_length=6,
            y_length=6,
            axis_config={"include_tip": True}
        )
        axes.shift(DOWN * 0.5)
        self.play(Create(axes))
        
        # Training data points
        np.random.seed(42)
        class1_points = [(2, 3), (3, 2), (2, 5), (4, 4)]
        class2_points = [(7, 7), (8, 6), (7, 8), (9, 7)]
        
        class1_dots = VGroup(*[
            Dot(axes.c2p(x, y), color=BLUE, radius=0.1)
            for x, y in class1_points
        ])
        class2_dots = VGroup(*[
            Dot(axes.c2p(x, y), color=RED, radius=0.1)
            for x, y in class2_points
        ])
        
        self.play(FadeIn(class1_dots), FadeIn(class2_dots))
        self.wait(0.5)
        
        # New point to classify
        new_point = Dot(axes.c2p(5, 5), color=YELLOW, radius=0.12)
        new_label = Text("?", font_size=24, color=YELLOW).next_to(new_point, UP)
        
        self.play(FadeIn(new_point), Write(new_label))
        self.wait(0.5)
        
        # Draw circle around new point (k=3)
        k = 3
        circle = Circle(
            radius=2.0,
            color=GREEN,
            stroke_width=3
        ).move_to(new_point.get_center())
        
        self.play(Create(circle))
        self.wait(0.5)
        
        # Highlight nearest neighbors
        nearest = [
            axes.c2p(4, 4),
            axes.c2p(7, 7),
            axes.c2p(8, 6)
        ]
        
        lines = VGroup(*[
            Line(new_point.get_center(), point, color=GREEN, stroke_width=2)
            for point in nearest
        ])
        
        self.play(Create(lines))
        self.wait(0.5)
        
        # Show k value
        k_label = Text("k = 3 neighbors", font_size=24, color=GREEN)
        k_label.next_to(axes, DOWN)
        self.play(Write(k_label))
        
        # Classify the point (majority vote)
        self.wait(1)
        self.play(
            new_point.animate.set_color(RED),
            new_label.animate.become(Text("Red!", font_size=24, color=RED).next_to(new_point, UP))
        )
        
        self.wait(2)
