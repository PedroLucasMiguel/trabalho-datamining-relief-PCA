import os

class ArffLib:
    def __init__(self, 
                 file_name:str, 
                 file_path:str, 
                 n_features:int) -> None:
        
        self.file_name = f"{file_name}.arff"
        self.file_path = file_path
        self.n_features = n_features
        self.n_appended_features = 0

        self.__file = None

        pass

    def create_file(self) -> None:
        self.__file = open(os.path.join(self.file_path, self.file_name), "w")

        # Iniciando a escrita do cabeçalho do arquivo
        self.__file.write(f"@RELATION {self.file_name}\n\n")

        for i in range(1, self.n_features+1):
            self.__file.write(f"@ATTRIBUTE x{i} REAL\n")

        self.__file.write("@ATTRIBUTE class {0,1}\n\n")
        self.__file.write("@DATA\n\n")

        print(f"ArffLib: O arquivo {self.file_name} foi criado em {self.file_path}. A stream está aberta!")

    def append_to_file(self, activations, label:int) -> None:
        for activation in activations:
            self.__file.write(f"{activation},")
        
        self.__file.write(f"{label}\n")
        self.n_appended_features += 1

    def reopen_file(self) -> None:
        if self.__file.closed:
            self.__file.open(os.path.join(self.file_path, self.file_name))
        else:
            raise Exception("ArffLib: File was not previously closed")

    def close_file(self) -> None:
        self.__file.close()