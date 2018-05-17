
import web
from PIL import Image
import io
import dbc.classifier as c

urls = (
   '/classify_dog', 'classify_dog',
   '/test', 'test'
)

class test:
    def GET(self):
        return 'test'

class classify_dog:
    def POST(self):
        web.header('Access-Control-Allow-Origin', '*')
        form = web.input(image = "img")
        image = Image.open(io.BytesIO(form.img))
        if c.cls is None:
            c.cls = c.Classifier()
        return c.cls.interact(image)

web.config.debug = False
app = web.application(urls, globals())
#app = app.wsgifunc()

if __name__ == "__main__":
    app.run()
