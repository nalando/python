from manimlib import *
class Example1(Scene):
    def construct(self):
        self.add(Circle())
        self.wait()
        S = Square()
        self.play(ShowCreation(S))
        self.play(S.move_to ,(2,2,0))
        T = Triangle().move_to((-2,-2,0))
        self.play(Transform(S,T))
        M = Tex("f(x)=x^2+\\sin x").move_to((-2,2,0))
        self.play(Write(M))
        self.play(Rotate(M,2*PI))
        self.play(Write(Text("Hello World!")))
if __name__ == "__main__":
    myfile = os.path.basename(__file__)
    os.system("manimgl -om "+myfile)
