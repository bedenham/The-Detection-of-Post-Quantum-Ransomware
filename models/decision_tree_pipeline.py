import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance
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

# -----------------------------
# Load data
# -----------------------------
data = pd.read_csv("data.csv", header=None)
data.columns = features + ["label"]  # assign names
X = data[features]
y = data["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train Decision Tree
# -----------------------------
tree = DecisionTreeClassifier(max_depth=4, random_state=42)  # limit depth for interpretability
tree.fit(X_train, y_train)

# Evaluate
y_pred = tree.predict(X_test)

with open("decisionTreeOutputs/dt.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    f.write(f"{classification_report(y_test, y_pred)}\n")
    f.write("\n")
    f.write("\n")
    f.write("\n")

    f.write("--------------------Tree Rules--------------------\n")
    tree_rules = export_text(tree, feature_names=[f"{features[i]}" for i in range(X.shape[1])])
    f.write("\nDecision Tree Rules:\n")
    f.write(tree_rules)
    f.write("\n")
    f.write("\n")
    f.write("\n")

    f.write("--------------------Top 20 Features: Gini importance (Mean decrease in impurity)--------------------\n")
    importances = tree.feature_importances_
    importance_df = pd.DataFrame({
        "feature": features,
        "importance": importances
    })
    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )
    f.write(importance_df.head(20).to_string())
    f.write("\n")
    f.write("\n")
    f.write("\n")
    top_gini = importance_df.head(20)
    plt.figure(figsize=(10, 8))
    plt.barh(top_gini["feature"], top_gini["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Top 20 Features - Gini Importance")
    plt.tight_layout()
    plt.savefig("decisionTreeOutputs/top20_gini.png")
    plt.close()

    f.write("--------------------Top 20 Features: Permutation Importance--------------------\n")
    result = permutation_importance(
        tree,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    perm_df = pd.DataFrame({
        "feature": features,
        "importance": result.importances_mean
    }).sort_values("importance", ascending=False)
    f.write(perm_df.head(20).to_string())
    f.write("\n")
    f.write("\n")
    f.write("\n")
    top_perm = perm_df.head(20)
    plt.figure(figsize=(10, 8))
    plt.barh(top_perm["feature"], top_perm["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Top 20 Features - Permutation Importance")
    plt.tight_layout()
    plt.savefig("decisionTreeOutputs/top20_permutation.png")
    plt.close()

    f.write("--------------------Top Features: Based on use and gini importance--------------------\n")
    used_indices = np.unique(tree.tree_.feature)
    used_indices = used_indices[used_indices >= 0]
    used_importance_df = pd.DataFrame({
        "feature": [features[i] for i in used_indices],
        "importance": [importances[i] for i in used_indices]
    }).sort_values("importance", ascending=False)
    f.write(used_importance_df.to_string())
    f.write("\n")
    f.write("\n")
    f.write("\n")
    plt.figure(figsize=(10, 8))
    plt.barh(used_importance_df["feature"], used_importance_df["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Features Used in Tree (Ranked by Gini)")
    plt.tight_layout()
    plt.savefig("decisionTreeOutputs/used_features_gini.png")
    plt.close()

    f.write("--------------------Cross Validation Accuracy--------------------\n")
    scores = cross_val_score(tree, X, y, cv=5)
    f.write(f"CV accuracy: {scores}\n")
    f.write(f"Mean CV accuracy: {scores.mean()}\n")
    f.write("\n")
    f.write("\n")
    f.write("\n")

ConfusionMatrixDisplay.from_estimator(tree, X_test, y_test)
plt.savefig("decisionTreeOutputs/confusionMatrix.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(20, 10))
plot_tree(
    tree,
    feature_names=features,
    class_names=[str(c) for c in tree.classes_],
    filled=True,
    rounded=True,
    fontsize=6
)
plt.title("Decision Tree Visualization")
plt.tight_layout()
plt.savefig("decisionTreeOutputs/tree_visualization.png", dpi=300)
plt.close()
