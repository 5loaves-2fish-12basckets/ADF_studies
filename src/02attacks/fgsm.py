from module.trainer import Trainer

def main():
    trainer = Trainer()
    trainer.train()
    trainer.test() #mnist, font digit 95
    trainer.attack()

if __name__ == '__main__':
    main()
