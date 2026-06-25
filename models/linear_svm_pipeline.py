import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

#features list in order: opcodeCounts,opCodePercentages,classCounts,classPercentages,unlistedCount,unlistedPercentage
features = [
"add", "addb", "addl", "addq", "adc", "adcl", "sub", "subl", "subq", "subw", "sbb", "sbbl", "mul", "mulq", "imul", "div", "divq", "idiv", "idivl", "inc", "dec", "neg", "negl", "negq", "and", "andb", "andl", "andq", "andw", "or", "orb", "orl", "orq", "orw", "xor", "xorl", "xorps", "not", "notq", "test", "testb", "testl", "testq", "testw", "bsf", "bsr", "bt", "btc", "btcq", "btr", "bts", "blsmsk", "bzhi", "tzcnt", "bswap", "shl", "shll", "shlq", "shr", "shrl", "sar", "rol", "ror", "shld", "shlx", "shrx", "sarx", "rorx", "cmp", "cmpb", "cmpl", "cmpq", "cmpw", "ucomisd", "comisd", "seta", "setae", "setb", "setbe", "sete", "setne", "setg", "setge", "setl", "setle", "seto", "cmova", "cmovae", "cmovb", "cmovbe", "cmove", "cmovne", "cmovg", "cmovge", "cmovl", "cmovle", "cmovs", "cmovns", "jmp", "ja", "jae", "jb", "jbe", "je", "jne", "jg", "jge", "jl", "jle", "jo", "jno", "js", "jns", "jp", "jrcxz", "call", "ret", "leave", "int3", "ud2", "mov", "movb", "movl", "movq", "movw", "movabs", "movbe", "lea", "xchg", "push", "pop", "movsbl", "movsbq", "movsbw", "movswl", "movswq", "movslq", "movzbl", "movzwl", "cltd", "cwtl", "cltq", "fldt", "fstp", "fstpt", "addsd", "subsd", "cvtsi2sd", "cvtsi2sdl", "cvttsd2si", "movsd", "movss", "pextrw", "movaps", "movups", "movapd", "movd", "movdqa", "movdqu", "movlpd", "movhpd", "movhps", "vmovd", "vmovq", "vmovdqa", "vmovdqu", "vmovntdq", "paddb", "paddw", "paddd", "paddq", "psubb", "psubw", "psubd", "psubq", "pmullw", "pmulhuw", "pmuludq", "vpaddb", "vpaddd", "pand", "por", "pxor", "vpand", "vpandn", "vpor", "vpxor", "pcmpeqb", "pcmpeqd", "pcmpgtb", "pcmpgtw", "pcmpgtd", "vpcmpeqb", "vpcmpeqd", "vpcmpgtb", "pshufb", "pshufd", "pshufhw", "pshuflw", "shufps", "shufpd", "punpcklbw", "punpcklwd", "punpckldq", "punpcklqdq", "punpckhbw", "punpckhwd", "punpckhdq", "psllw", "pslld", "psllq", "pslldq", "psrlw", "psrld", "psrlq", "psrldq", "psraw", "vpslld", "vpslldq", "vpsrld", "vpsrldq", "pinsrd", "pinsrw", "pmovmskb", "vpmovmskb", "pminub", "vpminub", "vpalignr", "vpbroadcastb", "vpbroadcastd", "vpshufb", "vinserti128", "vzeroupper", "aesenc", "aesenclast", "aesdec", "aesdeclast", "aeskeygenassist", "lock", "rep", "repz", "pause", "prefetcht0", "sfence", "cpuid", "rdtsc", "syscall", "xsave", "xrstor", "endbr64", "incsspq", "rdsspq", "rstorssp", "saveprevssp", "addr32", "data16", "cs", "local", "notrack", "nop", "nopl", "nopw", "andn", "packuswb", "pcmpistri",
"add%", "addb%", "addl%", "addq%", "adc%", "adcl%", "sub%", "subl%", "subq%", "subw%", "sbb%", "sbbl%", "mul%", "mulq%", "imul%", "div%", "divq%", "idiv%", "idivl%", "inc%", "dec%", "neg%", "negl%", "negq%", "and%", "andb%", "andl%", "andq%", "andw%", "or%", "orb%", "orl%", "orq%", "orw%", "xor%", "xorl%", "xorps%", "not%", "notq%", "test%", "testb%", "testl%", "testq%", "testw%", "bsf%", "bsr%", "bt%", "btc%", "btcq%", "btr%", "bts%", "blsmsk%", "bzhi%", "tzcnt%", "bswap%", "shl%", "shll%", "shlq%", "shr%", "shrl%", "sar%", "rol%", "ror%", "shld%", "shlx%", "shrx%", "sarx%", "rorx%", "cmp%", "cmpb%", "cmpl%", "cmpq%", "cmpw%", "ucomisd%", "comisd%", "seta%", "setae%", "setb%", "setbe%", "sete%", "setne%", "setg%", "setge%", "setl%", "setle%", "seto%", "cmova%", "cmovae%", "cmovb%", "cmovbe%", "cmove%", "cmovne%", "cmovg%", "cmovge%", "cmovl%", "cmovle%", "cmovs%", "cmovns%", "jmp%", "ja%", "jae%", "jb%", "jbe%", "je%", "jne%", "jg%", "jge%", "jl%", "jle%", "jo%", "jno%", "js%", "jns%", "jp%", "jrcxz%", "call%", "ret%", "leave%", "int3%", "ud2%", "mov%", "movb%", "movl%", "movq%", "movw%", "movabs%", "movbe%", "lea%", "xchg%", "push%", "pop%", "movsbl%", "movsbq%", "movsbw%", "movswl%", "movswq%", "movslq%", "movzbl%", "movzwl%", "cltd%", "cwtl%", "cltq%", "fldt%", "fstp%", "fstpt%", "addsd%", "subsd%", "cvtsi2sd%", "cvtsi2sdl%", "cvttsd2si%", "movsd%", "movss%", "pextrw%", "movaps%", "movups%", "movapd%", "movd%", "movdqa%", "movdqu%", "movlpd%", "movhpd%", "movhps%", "vmovd%", "vmovq%", "vmovdqa%", "vmovdqu%", "vmovntdq%", "paddb%", "paddw%", "paddd%", "paddq%", "psubb%", "psubw%", "psubd%", "psubq%", "pmullw%", "pmulhuw%", "pmuludq%", "vpaddb%", "vpaddd%", "pand%", "por%", "pxor%", "vpand%", "vpandn%", "vpor%", "vpxor%", "pcmpeqb%", "pcmpeqd%", "pcmpgtb%", "pcmpgtw%", "pcmpgtd%", "vpcmpeqb%", "vpcmpeqd%", "vpcmpgtb%", "pshufb%", "pshufd%", "pshufhw%", "pshuflw%", "shufps%", "shufpd%", "punpcklbw%", "punpcklwd%", "punpckldq%", "punpcklqdq%", "punpckhbw%", "punpckhwd%", "punpckhdq%", "psllw%", "pslld%", "psllq%", "pslldq%", "psrlw%", "psrld%", "psrlq%", "psrldq%", "psraw%", "vpslld%", "vpslldq%", "vpsrld%", "vpsrldq%", "pinsrd%", "pinsrw%", "pmovmskb%", "vpmovmskb%", "pminub%", "vpminub%", "vpalignr%", "vpbroadcastb%", "vpbroadcastd%", "vpshufb%", "vinserti128%", "vzeroupper%", "aesenc%", "aesenclast%", "aesdec%", "aesdeclast%", "aeskeygenassist%", "lock%", "rep%", "repz%", "pause%", "prefetcht0%", "sfence%", "cpuid%", "rdtsc%", "syscall%", "xsave%", "xrstor%", "endbr64%", "incsspq%", "rdsspq%", "rstorssp%", "saveprevssp%", "addr32%", "data16%", "cs%", "local%", "notrack%", "nop%", "nopl%", "nopw%", "andn%", "packuswb%", "pcmpistri%",
"integer_arithmetic", "logical_bitwise", "bit_manipulation", "shifts_rotates", "comparison_fp_scalar", "flag_set", "conditional_moves", "control_flow_branches", "calls_returns", "interrupts_traps", "data_movement_scalar", "sign_zero_extension", "stack_arithmetic_helpers", "floating_point_x87", "floating_point_sse_scalar", "simd_data_movement", "simd_integer_arithmetic", "simd_logical", "simd_comparison", "simd_shuffle_unpack", "simd_shifts", "simd_misc", "crypto_aes", "synchronization_hints", "cpu_info_timing", "syscalls", "cpu_state_save_restore", "cet_shadow_stack", "assembler_prefixes", "miscellaneous",
"integer_arithmetic%", "logical_bitwise%", "bit_manipulation%", "shifts_rotates%", "comparison_fp_scalar%", "flag_set%", "conditional_moves%", "control_flow_branches%", "calls_returns%", "interrupts_traps%", "data_movement_scalar%", "sign_zero_extension%", "stack_arithmetic_helpers%", "floating_point_x87%", "floating_point_sse_scalar%", "simd_data_movement%", "simd_integer_arithmetic%", "simd_logical%", "simd_comparison%", "simd_shuffle_unpack%", "simd_shifts%", "simd_misc%", "crypto_aes%", "synchronization_hints%", "cpu_info_timing%", "syscalls%", "cpu_state_save_restore%", "cet_shadow_stack%", "assembler_prefixes%", "miscellaneous%",
"unlistedCount", "unlistedPercentage"
]

# Load data
data = pd.read_csv("data.csv", header=None)

X = data.iloc[:, :-1].values  # features
y = data.iloc[:, -1].values   # class labels

# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Build pipeline
# -----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(
        penalty="l1",
        dual=False,
        C=0.5,
        multi_class="ovr",
        max_iter=5000,
        random_state=42,
        verbose=1
    ))
])

# -----------------------------
# Train
# -----------------------------
pipeline.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = pipeline.predict(X_test)

svm = pipeline.named_steps["svm"]

coef = svm.coef_
intercept = svm.intercept_

importance = np.sum(np.abs(svm.coef_), axis=0)
top_features = np.argsort(importance)[-20:]

with open("svmOutputs/svm.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    f.write(f"{classification_report(y_test, y_pred)}\n")
    f.write("\n")
    f.write("\n")
    f.write("\n")

    f.write("--------------------Non-zero Coefficient Counts--------------------\n")
    for i, class_label in enumerate(svm.classes_):
        nonzero = np.sum(coef[i] != 0)
        f.write(f"Class {class_label}: {nonzero} non-zero coefficients " f"out of {coef.shape[1]}\n")
    f.write("\n")
    f.write("\n")
    f.write("\n")

    f.write("--------------------Top Features Per Class--------------------\n")
    for i, class_label in enumerate(svm.classes_):
        f.write(f"Class {class_label}\n")
        coefs = svm.coef_[i]
        top_positive = np.argsort(coefs)[-10:]
        top_negative = np.argsort(coefs)[:10]
        f.write("  Strongest positive:\n")
        for idx in reversed(top_positive):
            f.write(f"    {features[idx]}: {coefs[idx]:.6f}\n")
        f.write("  Strongest negative:\n")
        for idx in top_negative:
            f.write(f"    {features[idx]}: {coefs[idx]:.6f}\n")
        f.write("\n")
        pos_sorted = top_positive[np.argsort(coefs[top_positive])]
        neg_sorted = top_negative[np.argsort(coefs[top_negative])]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.barh(
            [features[j] for j in neg_sorted],
            coefs[neg_sorted],
            color="red"
        )
        ax.barh(
            [features[j] for j in pos_sorted],
            coefs[pos_sorted],
            color="green"
        )
        ax.set_title(f"Top 10 Positive and Negative Features\nClass {class_label}")
        ax.set_xlabel("Coefficient Value")
        plt.tight_layout()
        plt.savefig(f"svmOutputs/class{class_label}TopFeatures.png",
                    dpi=300,
                    bbox_inches="tight")
        plt.close(fig)
    f.write("\n")
    f.write("\n")
    f.write("\n")
    
    f.write("--------------------Top 20 Globally Important Features--------------------\n")
    for idx in reversed(top_features):
        f.write(f"{features[idx]}: {importance[idx]:.6f}\n")
    f.write("\n")
    f.write("\n")
    f.write("\n")

    f.write("--------------------Cross Validation Accuracy--------------------\n")
    scores = cross_val_score(pipeline, X, y, cv=5)
    f.write(f"CV accuracy: {scores}\n")
    f.write(f"Mean CV accuracy: {scores.mean()}\n")
    f.write("\n")
    f.write("\n")
    f.write("\n")
    
    f.write("--------------------Coefficients--------------------\n")
    for class_idx, class_label in enumerate(svm.classes_):
        f.write(f"Class {class_label}\n")
        f.write(f"Intercept: {svm.intercept_[class_idx]}\n")
        for j, coef in enumerate(svm.coef_[class_idx]):
            f.write(f"  {features[j]}: {coef}\n")
        f.write("\n")

ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)
plt.savefig("svmOutputs/confusionMatrix.png", dpi=300, bbox_inches="tight")
plt.close()


top_idx = np.argsort(importance)[-20:]
plt.barh([features[i] for i in top_idx], importance[top_idx])
plt.title("Top 20 Feature Importances")
plt.savefig("svmOutputs/top20Features.png", dpi=300, bbox_inches="tight")
plt.close()

