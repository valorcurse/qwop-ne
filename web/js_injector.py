"""
This example shows two ways to redirect flows to another server.
"""
from mitmproxy import http
from mitmproxy import ctx

def response(flow: http.HTTPFlow) -> None:
    # pretty_host takes the "Host" header of the request into account,
    # which is useful in transparent mode where we usually only have the IP
    # otherwise.
    # ctx.log.info(flow.request.path)
    if flow.request.path == "/QWOP.min.js":
        ctx.log.info("Found QWOP file.")
        flow.request.path = "file:///home/marcelo/Repositories/qwop-ne/web/QWOP.min.js"
        ctx.log.info(flow.request.path)

        fileHandle = open("QWOP.min.js")
        contents = fileHandle.read()

        flow.response.code = 200
        flow.response.text = contents