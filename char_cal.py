

string = "Hello this is Sajad from Barcelona."

character = "Sajad"
start_ix = len(string.split(character)[0])
end_ix = len(character)
print(
    f'start: {start_ix} - end: {start_ix + end_ix}'
)
print(string[start_ix: start_ix+end_ix])