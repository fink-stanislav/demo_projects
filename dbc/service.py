
import web
from PIL import Image
import io
from dbc.classifier import classifier

urls = (
   '/classify_dog', 'classify_dog'
)

class classify_dog:

    def POST(self):
        form = web.input(image = "img")
        image = Image.open(io.BytesIO(form.img))
        return classifier.interact(image)

app = web.application(urls, globals())
app = app.wsgifunc()

#if __name__ == "__main__":
#    app.run()
