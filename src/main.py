from train import get_trainer


def main():
    trainer = get_trainer()
    trainer.train()


if __name__=='__main__':
    main()