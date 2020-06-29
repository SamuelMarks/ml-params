def camel_case(st, upper=False):
    output = ''.join(x for x in st.title() if x.isalnum())
    return getattr(output[0], 'upper' if upper else 'lower')() + output[1:]
