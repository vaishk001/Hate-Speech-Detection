# Label Guide - KRIXION Hate Speech Detection

This guide defines the three classification labels used in the KRIXION Hate Speech Detection system.

## Label Definitions

### Label 0: Normal

**Definition:** Non-offensive, neutral, or positive content that does not contain hate speech, personal attacks, or aggressive language.

**Characteristics:**

- Respectful communication
- Constructive feedback
- Neutral statements
- Positive messages
- Questions without aggression
- Factual information

**Examples:**

✅ **Normal (Label 0):**

```
"Good morning! Hope you have a great day."
"I disagree with your point, but I respect your opinion."
"Can you please share the project details?"
"This is a well-written article."
"Let's work together to solve this problem."
"Thank you for your help!"
"The weather is nice today."
```

**Hindi/Hinglish Examples:**

```
"Namaste! Aap kaise hain?"
"Yeh achha idea hai."
"Mujhe aapki baat samajh aayi."
```

**Edge Cases (Still Normal):**

```
"I don't like pineapple on pizza." (Personal preference)
"This movie was boring." (Critique without attack)
"I'm frustrated with this software." (Frustration without hate)
```

---

### Label 1: Offensive

**Definition:** Mildly aggressive, rude, or disrespectful language that includes personal attacks, insults, or aggressive tone but does not rise to the level of severe hate speech.

**Characteristics:**

- Personal insults ("idiot", "stupid", "moron")
- Dismissive language
- Mocking tone
- Mild aggression
- Profanity (non-hateful)
- Bullying behavior

**Examples:**

⚠️ **Offensive (Label 1):**

```
"You are an idiot for believing that."
"This is the stupidest idea I've ever heard."
"Stop being such a fool."
"You're a complete moron."
"Shut up, nobody asked you."
"You're so dumb, it's embarrassing."
"What a loser."
```

**Hindi/Hinglish Examples:**

```
"Tum bahut stupid ho."
"Chup kar, koi nahi sunna chahta."
"Tum ek bewakoof ho."
```

**Profanity (Offensive, not Hate):**

```
"This is f***ing ridiculous."
"What the hell is wrong with you?"
"Damn, you're annoying."
```

**Edge Cases:**

```
"You're wrong." → Normal (factual disagreement)
"You're an idiot and wrong." → Offensive (insult + disagreement)
"I hate this app." → Normal (criticism of object)
"I hate you." → Offensive (personal attack)
```

---

### Label 2: Hate

**Definition:** Severe hate speech that promotes violence, discrimination, or intense hostility against individuals or groups based on identity, characteristics, or affiliations.

**Characteristics:**

- Violent threats
- Death wishes
- Discrimination based on race, religion, gender, etc.
- Dehumanization
- Incitement to violence
- Extreme hostility
- Genocidal rhetoric

**Examples:**

🚫 **Hate (Label 2):**

```
"I will kill you."
"You should die."
"All [group] are trash and should be eliminated."
"I hate all [religion/race/gender] people."
"Let's burn down their homes."
"They don't deserve to live."
"[Group] are subhuman."
```

**Discriminatory Hate:**

```
"[Racial slur] should go back to their country."
"[Religious group] are terrorists."
"Women are inferior and should stay silent."
"[LGBTQ+ slur] are disgusting."
```

**Violent Threats:**

```
"I'll find you and hurt you."
"Someone should shoot them."
"Let's attack [group] tonight."
```

**Hindi/Hinglish Examples:**

```
"Main tumhe maar dunga." (I will kill you.)
"Sab [group] ko maarna chahiye." (All [group] should be killed.)
```

**Edge Cases:**

```
"I hate Mondays." → Normal (not directed at person)
"I hate you." → Offensive (personal but not violent)
"I will kill you." → Hate (violent threat)
```

---

## Label Decision Tree

```
Is the text aggressive or negative?
│
├─ NO → Label 0 (Normal)
│
└─ YES
   │
   └─ Does it contain violent threats, discrimination, or dehumanization?
      │
      ├─ YES → Label 2 (Hate)
      │
      └─ NO
         │
         └─ Does it contain insults, profanity, or personal attacks?
            │
            ├─ YES → Label 1 (Offensive)
            │
            └─ NO → Label 0 (Normal)
```

## Annotation Guidelines

### When Labeling as Normal (0):

1. No personal attacks
2. No aggressive language
3. Constructive or neutral tone
4. Respectful disagreement is OK
5. Criticism of ideas (not people) is OK

### When Labeling as Offensive (1):

1. Personal insults ("idiot", "stupid")
2. Dismissive/mocking tone
3. Profanity directed at person
4. Aggressive but not violent
5. Bullying behavior

### When Labeling as Hate (2):

1. Violent threats ("kill", "die")
2. Discrimination against groups
3. Dehumanization
4. Incitement to violence
5. Extreme hostility

## Ambiguous Cases

### Sarcasm

```
"Oh great, another genius idea." → Context-dependent
- If mocking: Offensive (1)
- If genuine: Normal (0)
```

### Cultural Context

```
"Pagal ho kya?" (Are you crazy?) → Offensive (1) in most contexts
- Among friends: Might be Normal (0)
- In argument: Offensive (1)
```

### Intensity Modifiers

```
"You're wrong." → Normal (0)
"You're so wrong." → Normal (0)
"You're f***ing wrong, idiot." → Offensive (1)
"You're wrong and should die." → Hate (2)
```

## Label Distribution (Current Dataset)

- **Normal (0):** 25 samples (37%)
- **Offensive (1):** 28 samples (41%)
- **Hate (2):** 15 samples (22%)

## Model Behavior

### Confidence Thresholds

- **High confidence (>0.8):** Reliable prediction
- **Medium confidence (0.5-0.8):** Review recommended
- **Low confidence (<0.5):** Human review required

### Common Misclassifications

1. **Normal → Offensive:** Profanity in neutral context
2. **Offensive → Hate:** Escalation detection
3. **Hate → Offensive:** Underestimation of severity

## User Feedback

If the model misclassifies:

1. Click "Incorrect" to flag
2. Click "Fix" to provide correct label
3. Feedback saved to `data/feedback.csv`
4. Used for future retraining

## Ethical Considerations

1. **Context Matters:** Same words can be Normal or Hate depending on context
2. **Cultural Sensitivity:** Slang and idioms vary by culture
3. **Reclamation:** Some groups reclaim slurs (context-dependent)
4. **Evolving Language:** New terms emerge constantly
5. **Human Review:** Always recommended for moderation decisions

## References

1. Hate Speech Detection Academic Literature
2. Content Moderation Best Practices
3. Online Harassment Research
4. Multilingual Offensive Language Corpora

## Contact

For labeling questions or disputes: [Your Email]
