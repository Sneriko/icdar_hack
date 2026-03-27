from train import TrOCRModule
import sys


if __name__ == "__main__":
    ckpt = sys.argv[1]
    module = TrOCRModule.load_from_checkpoint(ckpt)
    path = ckpt.replace(".ckpt", "")
    module.model.save_pretrained(path)
    module.processor.save_pretrained(path)
    module.processor.tokenizer.save_pretrained(path)
    print(f"Exported checkpoint '{ckpt}' to '{path}'")
