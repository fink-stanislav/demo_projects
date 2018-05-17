
import web
from PIL import Image
import io
from dbc.classifier import classifier

urls = (
   '/classify_dog', 'classify_dog',
   '/test', 'test'
)

class test:
    def GET(self):
        return 'test'

class classify_dog:
    def POST(self):
        form = web.input(image = "img")
        image = Image.open(io.BytesIO(form.img))
        return classifier.interact(image)

app = web.application(urls, globals())

if __name__ == "__main__":
    app.run()
