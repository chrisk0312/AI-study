# 클래스에서 상속이란, 물려주는 클래스(Parent Class, Super class)의 내용(속성과 메소드)을 
# 물려받는 클래스(Child class, sub class)가 가지게 되는 것입니다.
# 예를 들면 국가라는 클래스가 있고, 그것을 상속받은 한국, 일본, 중국, 미국 등의 클래스를 만들 수 있으며,
# 국가라는 클래스의 기본적인 속성으로 인구라는 속성을 만들었다면, 상속 받은 한국, 일본, 중국 등등의 클래스에서
# 부모 클래스의 속성과 메소드를 사용할 수 있음을 말합니다.
#자식클래스를 선언할때 소괄호로 부모클래스를 포함시킵니다.그러면 자식클래스에서는 부모클래스의 속성과 메소드는 기재하지 않아도 포함이 됩니다.

class Animal: # 동물이라는 클래스 정의
    def __init__(self, name): # 생성자 메소드
        self.name = name # 속성 name을 정의

    def speak(self): # 메소드 speak를 정의
        raise NotImplementedError("Subclass must implement this abstract method") # NotImplementedError를 발생시킴


class Dog(Animal): # 동물 클래스를 상속받는 Dog 클래스 정의
    def __init__(self, name, breed): # 생성자 메소드(breed 속성 추가)
        super().__init__(name)  # 부모클래스(동물)의 생성자 메소드 호출
        self.breed = breed # 속성 breed를 정의

    def speak(self): # 메소드 speak를 재정의
        return f"{self.name} says Woof!" # Woof!를 반환


class Cat(Animal):# 동물 클래스를 상속받는 Cat 클래스 정의
    def __init__(self, name, another_argument):# 생성자 메소드(another_argument 속성 추가)
        super().__init__(name)# 부모클래스(동물)의 생성자 메소드 호출
        self.another_argument = another_argument# 속성 another_argument를 정의

    def speak(self):  # 메소드 speak를 재정의
        return f"{self.name} says Meow!" # Meow!를 반환


A = Dog("강아지1호","강아지2호") # Dog 클래스의 인스턴스 생성
print(A.speak()) # 강아지1호 says Woof!

B = Cat("고양이1호", "고양이2호") # Cat 클래스의 인스턴스 생성
print(B.speak())  #고양이1호 says Meow!



