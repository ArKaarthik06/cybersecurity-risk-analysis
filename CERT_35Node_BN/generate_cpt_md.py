import cert_bn_builder

def generate_markdown():
    params = cert_bn_builder.CERTCPTBuilder.expert_params()['nodes']
    
    with open('results/CPT_Tables.md', 'w') as f:
        f.write('# Conditional Probability Tables (Expert Parameters)\n\n')
        f.write('These tables represent the expert-defined parameters for the 35-node CERT Bayesian Network.\n\n')
        
        # Layer 1: Evidence Priors
        f.write('## Layer 1: Observable Evidence (Priors)\n\n')
        f.write('| Node | P(Active=1) |\n')
        f.write('|---|---|\n')
        for node in cert_bn_builder.EVIDENCE_NODES:
            prior = params[node]['prior'][1]
            f.write(f'| {node} | {prior:.2f} |\n')
        f.write('\n')
        
        # Layer 2: Behavioral Noisy-OR
        f.write('## Layer 2: Behavioral Indicators (Noisy-OR)\n\n')
        f.write('| Node | Parents | Weights | Leak |\n')
        f.write('|---|---|---|---|\n')
        for node in cert_bn_builder.BEHAVIORAL_NODES:
            parents = params[node]['parents']
            weights = params[node]['noisy_or']['weights']
            leak = params[node]['noisy_or']['leak']
            p_str = ', '.join(parents)
            w_str = ', '.join([f'{w:.2f}' for w in weights])
            f.write(f'| {node} | {p_str} | {w_str} | {leak:.2f} |\n')
        f.write('\n')
        
        # Layer 3: Threat Hypotheses Noisy-OR
        f.write('## Layer 3: Threat Hypotheses (Noisy-OR)\n\n')
        f.write('| Node | Parents | Weights | Leak |\n')
        f.write('|---|---|---|---|\n')
        for node in cert_bn_builder.THREAT_NODES:
            parents = params[node]['parents']
            weights = params[node]['noisy_or']['weights']
            leak = params[node]['noisy_or']['leak']
            p_str = ', '.join(parents)
            w_str = ', '.join([f'{w:.2f}' for w in weights])
            f.write(f'| {node} | {p_str} | {w_str} | {leak:.2f} |\n')
        f.write('\n')
        
        # Layer 4: Risk Level CPT
        f.write('## Layer 4: Risk Level (Deterministic Mapping)\n\n')
        f.write('| DataExfiltration | UnauthorizedAccess | ITSabotage | IPTheft | P(Low) | P(Medium) | P(High) |\n')
        f.write('|---|---|---|---|---|---|---|\n')
        
        cpt = params[cert_bn_builder.RISK_NODE]['cpt']
        # Sort by number of threats active
        for k in sorted(cpt.keys(), key=lambda x: (sum(x), x)):
            probs = cpt[k]
            f.write(f'| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {probs[0]:.2f} | {probs[1]:.2f} | {probs[2]:.2f} |\n')
        
generate_markdown()
