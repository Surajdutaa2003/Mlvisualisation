from manim import *
import numpy as np

class LinearRegressionScene(Scene):
    def construct(self):
        axes = Axes(
            x_range=[0, 10],
            y_range=[0, 10],
            x_length=6,
            y_length=6
        )
        self.play(Create(axes))

        # Data points
        points = [
            Dot(axes.c2p(x, 0.8*x + np.random.uniform(-0.5, 0.5)))
            for x in range(1, 9)
        ]
        self.play(LaggedStart(*[FadeIn(p) for p in points]))

        # Regression line
        line = axes.plot(lambda x: 0.8*x, color=YELLOW)
        self.play(Create(line))
        
