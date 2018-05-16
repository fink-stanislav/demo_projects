
import web

urls = (
   '/test', 'test'
)

class test:

    def GET(self):
        return 'ey, kornew'


app = web.application(urls, globals())
app = app.wsgifunc()
