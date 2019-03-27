
def main():
    trianer = Trainer()
    trainer.train()
    trainer.test() #mnist, font digit 95
    trainer.attack()

if __name__ == '__main__':
    main()
