import sys
sys.path.append('../Engine')
import tornado.web
import tornado.ioloop
import cv2 as cv
from predicter import *


class uploadImgHandler(tornado.web.RequestHandler):
  def post(self):
    files = self.request.files["fileImage"]
    for f in files:
      fh = open(f"img/{f.filename}", "wb")
      fh.write(f.body)
      fh.close()
      imagePathForPrediction = f'../Server/img/{f.filename}'
      imagePath = f'./img/{f.filename}'
      image = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
      result = predictDiagnosis(image, imagePathForPrediction)
      diagnosis = 'There is no tumor.'
      if result == 1:
        diagnosis = 'There is a tumor.'
      print (diagnosis)
      self.render("index.html", display='inline', diagnosis=diagnosis)
  def get(self):
    self.render("index.html", display='none', diagnosis='none')

if (__name__ == "__main__"):
  app = tornado.web.Application([
    ("/", uploadImgHandler),
    ("/img/(.*)", tornado.web.StaticFileHandler, {'path': 'img'})
  ])

  app.listen(8080)
  print("Listening on port 8080")
  tornado.ioloop.IOLoop.instance().start()