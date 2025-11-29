import piexif
class EXIFChecker:
    def analyze(self,path):
        try: ex=piexif.load(path); tags=ex.get('Exif',{}); issues=[]; 
        except: return ['exif_error']; return issues
