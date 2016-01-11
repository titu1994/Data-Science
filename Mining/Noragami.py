from Mining import WebScrapping as ws

soup = ws.createScrapper("https://en.wikipedia.org/wiki/List_of_Noragami_episodes")

allepisodes = soup.find_all("td", {"class","summary"})

season1 = allepisodes[:12]
ovaseason = allepisodes[12:14]
season2 = allepisodes[14:26]

def addToList(eps, lst):
    for i, ep in enumerate(eps):
        title = ep.get_text()
        title = title.split("\n")
        title[0] = title[0].replace("\"", "")
        if len(title) > 1:
            title[1] = title[1].replace("\"", "")
        name = str(i+1) + ". " + title[0] + " (" + (title[1] if len(title) > 1 else "") + ")"
        lst.append(name)

season1episodes = ["Season 1:"]
addToList(season1, season1episodes)

ovas = ["OVAs:"]
addToList(ovaseason, ovas)

season2episodes = ["Season 2:"]
addToList(season2, season2episodes)

for x in season1episodes:
    print(x)

for x in season2episodes:
    print(x)

for x in ovas:
    print(x)