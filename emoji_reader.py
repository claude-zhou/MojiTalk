def get_emoji_codes(filename):
    emoji_dict = {}
    emoji_64_dict = {}

    with open(filename, encoding="utf-8") as f:
        line = f.readline()
        while line != "":
            if line.strip() == "" or line.startswith('#'):
                line = f.readline()
                continue
            if not(line.startswith('1') or line.startswith('2')):
                line = f.readline()
                continue
            # codes = line.split(';')[0].strip().split(' ')
            # codes = [chr(int(code, 16)) for code in codes]  # convert encoded sss to unicode character

            emoji, desc = line.strip().split('# ')[-1].split(' ', 1)
            emoji_dict[emoji] = desc
            if desc.endswith(" 64"):
                emoji_64_dict[emoji] = desc
            line = f.readline()

    return emoji_dict, emoji_64_dict

emoji_dict, emoji_64_dict = get_emoji_codes("emoji-test.txt")
emojis = list(emoji_dict.keys())
assert len(emoji_64_dict) == 64
emoji_64 = list(emoji_64_dict.keys())