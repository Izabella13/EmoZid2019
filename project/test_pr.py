from time import sleep
from pygame import mixer
from em_detect import em_7
import glob, cv2, dlib
win = dlib.image_window()
mixer.init()
emo=['Радость', 'Удивление', 'Грусть', 'Злость', 'Отвращение', 'Презрение', 'Страх']
sound=['1.mp3', '2.mp3', '3.mp3', '4.mp3', '5.mp3', '6.mp3', '7.mp3']
pics = ['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg']
for i, em in enumerate(emo):
    em_res=0
    while em_res<0.7:
        img = dlib.load_rgb_image(pics[i])
        win.clear_overlay()
        win.set_image(img)
        mixer.music.load(sound[i])
        mixer.music.play()
        print("Покажите нам "+ em)
        em_res=em_7(i+1)
        if em_res>=0.7:
            mixer.music.load('8.mp3')
            mixer.music.play()
            print("Молодец")
            sleep(2)
        else:
            mixer.music.load('9.mp3')
            mixer.music.play()
            print("Попробуй еще")
            sleep (2)
            
            
       