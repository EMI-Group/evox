import polib

pattern = "Duplicate implicit target name"

if __name__ == "__main__":
    po = polib.pofile("source/locale/zh_CN/LC_MESSAGES/docs.po")
    i = 0
    while i < len(po):
        entry = po[i]
        if pattern in entry.msgid:
            po.remove(entry)
        else:
            i += 1

    po.save("source/locale/zh_CN/LC_MESSAGES/docs.po")
