from . import model

def get_model(name):
  return {
      'UGATIT': model.UGATIT,
  }[name]
