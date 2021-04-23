import requests
import pdfkit
import imgkit

def get_infobox_html():
    resp = requests.get("http://en.wikipedia.org/w/api.php?action=query&prop=revisions&rvprop=content&format=json&titles=Tom%C3%A1%C5%A1_Garrigue_Masaryk&rvsection=0&rvparse").json()
    page_one = next(iter(resp['query']['pages'].values()))
    revisions = page_one.get('revisions', [])
    html = next(iter(revisions[0].values()))
    print(html)

options = {
    'encoding':'utf-8',
    'enable-local-file-access': None
}
pdfkit.from_file('html_page.html', 'out.pdf', options=options)
imgkit.from_file('html_page.html', 'out.jpg', options=options)