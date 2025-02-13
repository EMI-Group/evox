import polib

pattern = "Duplicate implicit target name"

if __name__ == "__main__":
    po = polib.pofile("docs/source/locale/zh/LC_MESSAGES/docs.po")
    i = 0
    while i < len(po):
        entry = po[i]
        if pattern in entry.msgid:
            po.remove(entry)
        else:
            i += 1

    po.save("docs/source/locale/zh/LC_MESSAGES/docs.po")
