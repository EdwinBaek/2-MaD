import pefile
import os
import pydasm
import sys

def run_opcode_extractor(origin_path, Dataset_path):
    print("extract start!")
    with open(origin_path + '/complete.txt', 'r') as f:
        check_list = []
        for line in f:
            check_list.append(line.strip())
        f.close()

    # extract opcode and APIcalls at trainset
    for file in os.listdir(Dataset_path):
        current_file = os.path.join(Dataset_path, file)  # Address of Dataset to analysis
        if not current_file in check_list:
            try:
                pe = pefile.PE(current_file)  # PE open
                # opcode list extract
                open_exe = open(current_file, 'rb')  # open dataset file
                data = open_exe.read()
                EntryPoint = pe.OPTIONAL_HEADER.AddressOfEntryPoint
                raw_size = pe.sections[0].SizeOfRawData
                EntryPoint_va = EntryPoint + pe.OPTIONAL_HEADER.ImageBase

                # Start disassembly at the EP
                offset = EntryPoint
                Endpoint = offset + raw_size

                # Loop until the end of the .text section
                f_opcode = open(origin_path + '/opcode/%s.txt' % file, 'w')  # open text file to write opcode

                while offset < Endpoint:
                    # Get the first instruction
                    i = pydasm.get_instruction(data[offset:], pydasm.MODE_32)
                    if not i:
                        break

                    # Print a string representation if the instruction
                    opcodes = pydasm.get_mnemonic_string(i, pydasm.FORMAT_INTEL)
                    f_opcode.write(opcodes + '\n')

                    # Go to the next instruction
                    offset += int(i.length)
                f_opcode.close()

                # API list extract
                API_list = []
                print("file = " + file)
                f_api = open(origin_path + '/api/%s.txt' % file, 'w')

                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    for API in entry.imports:
                        API_list.append(API.name)
                        f_api.write(str(API.name) + '\n')
                f_api.close()
                del API_list

                complete = open(origin_path + '/complete.txt', 'a')
                complete.write(str(current_file) + '\n')
                complete.close()

            except:
                print("Error! about : " + file)
                pass
        else:
            pass

    del check_list
    print("extract Done!")


if __name__ == '__main__':
    trainset_path = 'C://Users/static/Desktop/dataset/trainset/trainSet'
    train_origin_path = 'C://Users/static/Desktop/dataset/trainset'

    preset_path = 'C://Users/static/Desktop/dataset/preset/preSet'
    pre_origin_path = 'C://Users/static/Desktop/dataset/preset'

    finalset1_path = 'C://Users/static/Desktop/dataset/finalset1/finalSet1'
    final1_origin_path = 'C://Users/static/Desktop/dataset/finalset1'

    finalset2_path = 'C://Users/static/Desktop/dataset/finalset2/finalSet2'
    final2_origin_path = 'C://Users/static/Desktop/dataset/finalset2'

    clear = lambda: os.system('clear')
    clear()

    run_opcode_extractor(train_origin_path, trainset_path)
    clear()
    run_opcode_extractor(pre_origin_path, preset_path)
    clear()
    run_opcode_extractor(final1_origin_path, finalset1_path)
    clear()
    run_opcode_extractor(final2_origin_path, finalset2_path)
    clear()

    current_file = 'D://Program Files/Nox64/bin/Nox.exe'
    #current_file = './2016112642.exe'
    pe = pefile.PE(current_file)  # PE open
    # opcode list extract
    open_exe = open(current_file, 'rb')  # open dataset file
    data = open_exe.read()
    EntryPoint = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    raw_size = pe.sections[0].SizeOfRawData
    EntryPoint_va = EntryPoint + pe.OPTIONAL_HEADER.ImageBase

    # Start disassembly at the EP
    offset = EntryPoint
    Endpoint = offset + raw_size

    # Loop until the end of the .text section
    f_opcode = open('./optest.txt', 'w')  # open text file to write opcode

    while offset < Endpoint:
        # Get the first instruction
        i = pydasm.get_instruction(data[offset:], pydasm.MODE_32)

        if not i:
            break

        # Print a string representation if the instruction
        opcodes = pydasm.get_mnemonic_string(i, pydasm.FORMAT_INTEL)
        aa0 = pydasm.get_operand_string(i, 0, pydasm.FORMAT_INTEL, offset)
        aa1 = pydasm.get_operand_string(i, 1, pydasm.FORMAT_INTEL, offset)
        aa2 = pydasm.get_operand_string(i, 2, pydasm.FORMAT_INTEL, offset)
        f_opcode.write(opcodes + ' ' + str(aa0) + ' ' + str(aa1) + ' ' + str(aa2) + '\n')

        # Go to the next instruction
        offset += int(i.length)
    f_opcode.close()