#import os
#os.environ['KIVY_IMAGE'] = 'pil,sdl2'
#os.environ['KIVY_VIDEO'] = 'ffpyplayer' 

import kivy
from kivy.app import App #'App' allows us to do all the graphics and creates a window and do low level stuff for us
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput


class MyGrid(GridLayout):
	def __init__(self, **kwargs): #'**' means we can handle infinite amount of kwargs
		super(MyGrid, self).__init__(**kwargs) #calling constructor of 'GridLayout'
		self.cols = 2
		self.add_widget(Label(text = "Name: "))
		self.name = TextInput(multiline = False) #since, here we are aiming for ony one line
		self.add_widget(self.name)

		self.add_widget(Label(text = "Last Name: "))
		self.lastName = TextInput(multiline = False) #since, here we are aiming for ony one line
		self.add_widget(self.lastName)

		self.add_widget(Label(text = "Email: "))
		self.email = TextInput(multiline = False) #since, here we are aiming for ony one line
		self.add_widget(self.email)




class MyApp(App):
	def build(self):
		#return Label(text = "Hi! This is Satyam.")
		return MyGrid()

if __name__ == "__main__":
	MyApp().run()