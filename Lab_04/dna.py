import random

DNALettersDict = {0: "A", 1: "T", 2: "C", 3: "G"}

#===============================================================================================
#Generate DNA sequence for the test
#===============================================================================================
OriginalDNALength = 1000

OriginalDNASeq = [DNALettersDict[random.randint(0, 3)] for i in range(OriginalDNALength)]

#print(OriginalDNASeq)

#===============================================================================================
#Generate Fragments
#===============================================================================================
NumberOfFragments = 1000
MinLength = 50
MaxLength = 400

def SelFragment(DNASeq):
    startPos = random.randint(0, OriginalDNALength-1-MinLength)
    length = random.randint(MinLength, MaxLength)
    return DNASeq[startPos:startPos+length]

ListOfFragments = [ SelFragment(OriginalDNASeq) for i in range(NumberOfFragments)]

#===============================================================================================
#Reconstruct DNA from fragments
#===============================================================================================
MINIMAL_OVERLAP = 10

def get_overlap(s1, s2):
    overlap = ""
    for i in range(1, (len(s1) + 1)):
        if s2[:i] == s1[-i:] and len(s2[:i]) > len(overlap):
            overlap = s2[:i]
            
    return overlap
    

def get_max_overlap(s1, s2):    
    overlap_s1_s2 = get_overlap(s1, s2)
    overlap_s2_s1 = get_overlap(s2, s1)
    return (overlap_s2_s1, s2, s1) if len(overlap_s2_s1) > len(overlap_s1_s2) else (overlap_s1_s2, s1, s2)


def remove_fragment_from_list(list_of_fragments, fragment):
    try:
        list_of_fragments.remove(fragment)
    except:
        pass


def find_two_most_common_fragments(list_of_fragments):
    max_overlap = ("", "", "")
    to_delete = []
    list_of_fragments.sort(key=len, reverse=True)
    for i in range(len(list_of_fragments) - 1):
        for j in range(i + 1, len(list_of_fragments)):
            shorter = list_of_fragments[j]
            longer = list_of_fragments[i]
            if shorter in longer:   # if shorter sequence is contained by the longer one - delete it - we've got it already
                to_delete.append(shorter)
                continue
            overlap = get_max_overlap(list_of_fragments[i], list_of_fragments[j])
            if len(overlap[0]) > len(max_overlap[0]):
                max_overlap = overlap
                
    for d in to_delete:
        remove_fragment_from_list(list_of_fragments, d)
        
    print("Overlap length: " + str(len(max_overlap[0])))
    
    if len(max_overlap[0]) < MINIMAL_OVERLAP:
        return "", ""

    return max_overlap[1], max_overlap[2]


def combine_fragments(f1, f2):
    overlap, s1, s2 = get_max_overlap(f1, f2)
    overlap_idx = s1.find(overlap)
    return s1[:overlap_idx] + s2
    

def reconstruct_dna(list_of_fragments):
    prev = 0
    while(len(list_of_fragments) > 1):
        print("Fragments: " + str(len(list_of_fragments)))
        if prev == len(list_of_fragments):
            break
        
        prev = len(list_of_fragments)
        f1, f2 = find_two_most_common_fragments(list_of_fragments)
        remove_fragment_from_list(list_of_fragments, f1)
        remove_fragment_from_list(list_of_fragments, f2)
        combined = combine_fragments(f1, f2)
        if combined:
            list_of_fragments.append(combined)


ListOfFragments = [''.join(f) for f in ListOfFragments] # list of list of chars -> list of strings

reconstruct_dna(ListOfFragments)

OriginalDNASeq = "".join(OriginalDNASeq)
reconstructed_seq = max(ListOfFragments, key=len)
print("Original DNA: " + OriginalDNASeq)
print("Original length: " + str(len(OriginalDNASeq)))
print("Reconstucted: " + str(ListOfFragments))
print("Reconstructed longest sequence: " + reconstructed_seq)
print("Reconstructed longest sequence length: " + str(len(reconstructed_seq)))

print("Is reconstructed DNA substring of original DNA? - " + str(reconstructed_seq in OriginalDNASeq))
print("Is reconstructed DNA equal to original DNA? - " + str(reconstructed_seq == OriginalDNASeq))