"""Low-cost input failures/collisions: no trained-model downloads or inference."""
import tempfile,unittest
from pathlib import Path
from PIL import Image
from restoration.pipeline import collect

class InputTests(unittest.TestCase):
    def test_jfif_webp_and_non_image_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);im=Image.new('RGB',(32,48),(100,120,150))
            im.save(root/'ảnh có khoảng trắng.jfif',format='JPEG');im.save(root/'photo.webp',format='WEBP')
            (root/'notes.txt').write_text('Not an image');rows,skipped,failed=collect(root)
            self.assertEqual(len(rows),2);self.assertEqual(len(skipped),1);self.assertFalse(failed)
            self.assertEqual({r['format'] for r in rows},{'JPEG','WEBP'})
    def test_same_stem_different_formats_kept_as_two_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);im=Image.new('RGB',(32,32));im.save(root/'photo.png');im.save(root/'photo.jpg')
            rows,_,failed=collect(root);self.assertEqual(len(rows),2);self.assertFalse(failed)
    def test_corrupt_image_and_tiny_image_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'bad.jpg').write_bytes(b'broken JPEG');Image.new('RGB',(2,2)).save(root/'tiny.png')
            rows,_,failed=collect(root);self.assertFalse(rows);self.assertEqual(len(failed),2)
    def test_subfolders_not_recursively_scanned(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'nested').mkdir();Image.new('RGB',(32,32)).save(root/'nested/photo.png')
            rows,_,_=collect(root);self.assertFalse(rows)

if __name__=='__main__':unittest.main()
