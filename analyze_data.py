import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import shutil
from matplotlib.lines import Line2D


def classify_conflict(left_emo, right_emo):

    emos = {left_emo, right_emo}
    if emos == {'positive', 'negative'}:
        return 'high'
    else:
        return 'low'


N_STEPS = 101


def calculate_auc(trial_data, n_steps=N_STEPS):

    x = trial_data['x'].values.astype(float)
    y = trial_data['y'].values.astype(float)
    t = trial_data['t_rel'].values.astype(float)
    if len(x) < 5:
        return 0.0
    if t[-1] <= t[0]:
        return 0.0

    s = (t - t[0]) / (t[-1] - t[0])
    grid = np.linspace(0.0, 1.0, n_steps)
    xn = np.interp(grid, s, x)
    yn = np.interp(grid, s, y)

    start = np.array([xn[0], yn[0]])
    end = np.array([xn[-1], yn[-1]])
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return 0.0
    line_unitvec = line_vec / line_len

    side = np.sign(end[0] - start[0])
    if side == 0:
        side = 1.0

    signed_dev = []
    for xi, yi in zip(xn, yn):
        p_vec = np.array([xi, yi]) - start
        cross = line_unitvec[0] * p_vec[1] - line_unitvec[1] * p_vec[0]
        signed_dev.append(side * cross)

    return float(np.trapezoid(signed_dev, grid))


def bimodality_coefficient(values):

    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    if n < 4:
        return np.nan
    sd = v.std(ddof=1)
    if sd == 0:
        return np.nan
    z = (v - v.mean()) / sd
    g1 = np.mean(z ** 3)
    g2 = np.mean(z ** 4) - 3.0
    return (g1 ** 2 + 1.0) / (g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def run_advanced_analysis(base_data_path="data"):

    session_dirs = [d for d in glob.glob(os.path.join(base_data_path, "*")) if os.path.isdir(d)]

    for session_path in session_dirs:
        session_name = os.path.basename(session_path)
        print(f"\nProcessing session: {session_name}")
        res_p = os.path.join(session_path, 'results.csv')
        traj_p = os.path.join(session_path, 'trajectories.csv')
        if not os.path.exists(res_p) or not os.path.exists(traj_p):
            continue

        df = pd.read_csv(res_p)
        df_t = pd.read_csv(traj_p)

        df_t = df_t[~((df_t['x'] == 0) & (df_t['y'] == 0) & (df_t['t_rel'] > 0.05))]

        base_rt = df['base_rt'].iloc[0]


        anomaly_path = os.path.join(session_path, 'anomaly_trials')
        normal_path = os.path.join(session_path, 'normal_trials')
        for p in [anomaly_path, normal_path]:
            if os.path.exists(p):
                shutil.rmtree(p)
            os.makedirs(p)

        sns.set_theme(style="whitegrid")

        trial_stats = []
        test_df_t = df_t[df_t['type'] != 'CALIBRATION']

        for n in test_df_t['n'].unique():
            t_data = test_df_t[test_df_t['n'] == n].sort_values('t_rel')
            if len(t_data) < 5:
                continue
            auc = calculate_auc(t_data)

            xs = t_data['x'].values
            committed = np.sign(xs[np.abs(xs) > 20])
            n_crossings = int((np.diff(committed) != 0).sum()) if committed.size > 1 else 0

            info_rows = df[df['n'] == n]
            if info_rows.empty:
                continue
            info = info_rows.iloc[0]

            trial_stats.append({
                'n': n,
                'type': info['type'],
                'auc': auc,
                'flip': n_crossings,
                'rt': info['rt'],
                'choice': info['choice'],
                'conflict': classify_conflict(info['emo_l'], info['emo_r']),
                'chosen_emo': info['chosen_emo'],
                'color': info.get('chosen_color', 'N/A')
            })

        df_d = pd.DataFrame(trial_stats)

        emo_df = df[df['chosen_emo'].isin(['positive', 'neutral', 'negative'])]
        if not emo_df.empty:
            emo_counts = emo_df['chosen_emo'].value_counts(normalize=True) * 100
            emo_counts = emo_counts.reindex(['positive', 'neutral', 'negative']).fillna(0)

            plt.figure(figsize=(6, 5))
            ax = sns.barplot(
                x=emo_counts.index,
                y=emo_counts.values,
                hue=emo_counts.index,
                palette={
                    'positive': '#4CAF50',
                    'neutral': '#9E9E9E',
                    'negative': '#F44336'
                },
                legend=False
            )

            plt.ylabel('Choice frequency (%)')
            plt.xlabel('Emotional category')
            plt.title('Emotion Choice Frequency')

            for i, v in enumerate(emo_counts.values):
                ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

            plt.ylim(0, 100)
            plt.savefig(os.path.join(session_path, 'emotion_choice_frequency.png'), dpi=150)
            plt.close()


        if 'type' in df_d.columns:
            main_d = df_d[~df_d['type'].isin(['SWAP', 'CALIBRATION', 'SRI'])]
        else:
            main_d = df_d

        auc_high_vals = main_d[main_d['conflict'] == 'high']['auc']
        auc_low_vals = main_d[main_d['conflict'] == 'low']['auc']
        auc_high = auc_high_vals.mean()
        auc_low = auc_low_vals.mean()

        dAUC = auc_high - auc_low

        DCI = (auc_high - auc_low) / auc_low if (pd.notna(auc_low) and auc_low != 0) else np.nan

        BC_high = bimodality_coefficient(auc_high_vals.values)


        sw = df[df['type'] == 'SWAP']
        sw_valid = sw[sw['is_swap'].isin([0, 1])]
        RSI = (sw_valid['is_swap'] == 0).mean() if not sw_valid.empty else np.nan

        FLIP = df_d['flip'].mean() if not df_d.empty else np.nan

        avg_auc = df_d['auc'].mean() if not df_d.empty else 0
        auc_abs_ref = df_d['auc'].abs().mean() if not df_d.empty else 0

        if 'chosen_color' in df.columns:
            all_appeared_colors = pd.concat([df['color_l'], df['color_r']])
            appearance_counts = all_appeared_colors.value_counts()

            choice_counts = df[~df['chosen_color'].isin(['none', 'error', 'N/A'])]['chosen_color'].value_counts()

            min_appearances = 3
            valid_colors = appearance_counts[appearance_counts >= min_appearances].index

            color_preference_pct = (choice_counts / appearance_counts * 100).fillna(0)
            color_preference_pct = color_preference_pct[color_preference_pct.index.isin(valid_colors)].sort_values(
                ascending=False)

            if not color_preference_pct.empty:
                plt.figure(figsize=(10, 6))
                pal = {c: c.lower() if c.lower() != 'white' else '#f0f0f0' for c in color_preference_pct.index}

                ax = sns.barplot(x=color_preference_pct.index, y=color_preference_pct.values,
                                 hue=color_preference_pct.index, palette=pal, legend=False)

                plt.axhline(50, color='black', linestyle='--', alpha=0.5, label='Random (50%)')
                plt.title(
                    f'Net color preference\n(% of choices from total presentations, min {min_appearances} appearances)')
                plt.ylabel('Preference, %')

                for i, c in enumerate(color_preference_pct.index):
                    v = color_preference_pct[c]
                    n_chosen = choice_counts.get(c, 0)
                    n_appeared = appearance_counts[c]
                    ax.text(i, v + 2, f'{int(v)}%\n({n_chosen}/{n_appeared})',
                            ha='center', fontweight='bold', fontsize=9)

                plt.legend()
                plt.savefig(os.path.join(session_path, 'color_preference.png'), dpi=150)
                plt.close()

            plt.figure(figsize=(10, 6))
            df_rt_color = df[~df['chosen_color'].isin(['none', 'error', 'N/A']) & (df['rt'] > 0)].copy()

            if not df_rt_color.empty:
                avg_rt_color = df_rt_color.groupby('chosen_color')['rt'].mean().sort_values()
                pal_rt = {c: c.lower() if c.lower() != 'white' else '#f0f0f0' for c in avg_rt_color.index}

                ax_rt = sns.barplot(x=avg_rt_color.index, y=avg_rt_color.values,
                                    hue=avg_rt_color.index, palette=pal_rt, legend=False)

                plt.axhline(base_rt, color='red', linestyle='--', alpha=0.6,
                            label=f'Base RT ({int(base_rt)}ms)')

                for i, v in enumerate(avg_rt_color.values):
                    ax_rt.text(i, v + 5, f'{int(v)}ms', ha='center', fontweight='bold')

                plt.title('Mean RT by color')
                plt.ylabel('RT (ms)')
                plt.legend()
                plt.savefig(os.path.join(session_path, 'rt_by_color.png'), dpi=150)
                plt.close()

        if not sw.empty:
            plt.figure(figsize=(6, 6))
            sc = sw['is_swap'].value_counts()
            plt.pie([sc.get(0, 0), sc.get(1, 0)], labels=['Consistent', 'Switched'], autopct='%1.1f%%',
                    colors=['#2196F3', '#FFC107'])
            plt.title('SWAP Test Consistency')
            plt.savefig(os.path.join(session_path, 'swap_consistency.png'))
            plt.close()

        plt.figure(figsize=(12, 5))
        sns.lineplot(data=df[df['rt'] > 0], x='n', y='rt', color='gray', alpha=0.3)
        sns.scatterplot(data=df[df['rt'] > 0], x='n', y='rt', hue='type', s=80, palette='bright')
        plt.axhline(base_rt, color='red', linestyle='--', label=f'Calibration ({int(base_rt)}ms)')
        plt.title('Reaction time (RT)')
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.savefig(os.path.join(session_path, 'rt_dynamics.png'), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 5))
        if 'conflict' in df_d.columns:
            sns.barplot(data=df_d, x='n', y='auc', hue='conflict', palette='viridis', dodge=False)
        else:
            sns.barplot(data=df_d, x='n', y='auc', color='steelblue')
        plt.axhline(avg_auc, color='black', linestyle=':', alpha=0.5, label='Average')
        plt.title('Cognitive effort index (AUC)')
        plt.legend()
        plt.savefig(os.path.join(session_path, 'auc_profile.png'))
        plt.close()

        for _, row in df_d.iterrows():
            is_anomaly = row['flip'] >= 1 or abs(row['auc']) > auc_abs_ref * 1.5
            curr_path = anomaly_path if is_anomaly else normal_path
            t_data = test_df_t[test_df_t['n'] == row['n']].sort_values('t_rel')

            fig, ax = plt.subplots(figsize=(8, 8))
            START_POS = (0, -500)
            click_x, click_y = t_data['x'].iloc[-1], t_data['y'].iloc[-1]

            ax.grid(True, linestyle=':', alpha=0.6)
            ax.axvline(0, color='black', alpha=0.2, lw=1)
            ax.axhline(0, color='black', alpha=0.2, lw=1)
            ax.add_patch(plt.Rectangle((-900, -100), 400, 400, color='gray', alpha=0.05))
            ax.add_patch(plt.Rectangle((500, -100), 400, 400, color='gray', alpha=0.05))

            ax.plot([START_POS[0], click_x], [START_POS[1], click_y], 'g--', alpha=0.3, label='Perfect route',
                    zorder=1)
            ax.plot(START_POS[0], START_POS[1], 'ro', markersize=10, label='Start (0, -500)', zorder=5,
                    markeredgecolor='black')
            sc = ax.scatter(t_data['x'], t_data['y'], c=t_data['t_rel'], cmap='autumn', s=35, zorder=4,
                            edgecolor='white', linewidth=0.3)
            ax.plot(t_data['x'], t_data['y'], 'k-', alpha=0.2, lw=1)
            ax.plot(click_x, click_y, 'go', markersize=10, label='Click (END)', zorder=6, markeredgecolor='black')

            cbar = plt.colorbar(sc)
            cbar.set_label('Time (ms)', rotation=270, labelpad=15)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

            ax.set_title(f"Trial #{row['n']} ({row['color']})\nAUC: {int(row['auc'])} | RT: {int(row['rt'])}ms")
            ax.set_xlim(-960, 960)
            ax.set_ylim(-600, 600)
            plt.savefig(os.path.join(curr_path, f"trial_{row['n']}_trace.png"), bbox_inches='tight')
            plt.close()



        if pd.notna(dAUC):
            verdict = "LCA (Model 2)" if dAUC > 0 else "minimal DDM (Model 1)"
            verdict_short = "-> LCA" if dAUC > 0 else "-> DDM"
        else:
            verdict, verdict_short = "n/a", ""

        def _headroom(ax, frac=0.20):
            lo, hi = ax.get_ylim()
            pad = frac * ((hi - lo) or 1.0)
            ax.set_ylim(lo - (pad if lo < 0 else 0), hi + pad)

        def _annot(ax, bars, fmt="{:.1f}"):
            lo, hi = ax.get_ylim()
            span = (hi - lo) or 1.0
            for b in bars:
                h = b.get_height()
                if np.isnan(h):
                    ax.text(b.get_x() + b.get_width() / 2, lo + 0.05 * span, 'n/a',
                            ha='center', va='bottom', fontsize=9, color='gray')
                    continue
                up = h >= 0
                ax.text(b.get_x() + b.get_width() / 2, h + (0.03 if up else -0.03) * span,
                        fmt.format(h), ha='center', va='bottom' if up else 'top',
                        fontsize=9, fontweight='bold')

        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        fig.suptitle(f'Decision Dynamics Profile — Session: {session_name}',
                     fontsize=13, fontweight='bold')

        ax = axes[0, 0]
        b = ax.bar(['high\n(Pos-Neg)', 'low\n(Neutral-X)'], [auc_high, auc_low],
                   color=['#b8412f', '#2e6e9e'], width=0.6, edgecolor='black', linewidth=0.6)
        ax.axhline(0, color='black', lw=1)
        ax.set_title('Signed AUC means', fontsize=10, fontweight='bold')
        ax.set_ylabel('AUC (px; toward competitor +)', fontsize=9)
        _headroom(ax); _annot(ax, b)

        ax = axes[0, 1]
        b = ax.bar(['ΔAUC'], [dAUC],
                   color=('#3a8a5f' if (pd.notna(dAUC) and dAUC > 0) else '#8a93a0'),
                   width=0.5, edgecolor='black', linewidth=0.6)
        ax.axhline(0, color='black', lw=1)
        ax.set_title(f'ΔAUC = high − low  {verdict_short}', fontsize=10, fontweight='bold')
        ax.set_ylabel('px (primary contrast)', fontsize=9)
        _headroom(ax); _annot(ax, b)

        ax = axes[0, 2]
        b = ax.bar(['DCI'], [DCI], color='#c79a3a', width=0.5, edgecolor='black', linewidth=0.6)
        ax.axhline(0, color='black', lw=1)
        ax.set_title('DCI = ΔAUC / AUC_low', fontsize=10, fontweight='bold')
        ax.set_ylabel('ratio (fragile if AUC_low<0)', fontsize=9)
        _headroom(ax); _annot(ax, b, "{:.2f}")

        ax = axes[1, 0]
        b = ax.bar(['RSI'], [RSI], color='#4c72b0', width=0.5, edgecolor='black', linewidth=0.6)
        ax.axhline(0.5, color='red', ls='--', alpha=0.7, lw=1.2)
        ax.text(0.45, 0.5, 'chance 0.5', color='red', fontsize=8, va='bottom', ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_title('RSI (choice stability)', fontsize=10, fontweight='bold')
        ax.set_ylabel('proportion maintained', fontsize=9)
        _annot(ax, b, "{:.2f}")

        ax = axes[1, 1]
        b = ax.bar(['FLIP'], [FLIP], color='#bdbdbd', width=0.5, edgecolor='black', linewidth=0.6)
        ax.axhline(0, color='black', lw=1)
        ax.set_title('FLIP (exploratory — excluded)', fontsize=10, fontweight='bold')
        ax.set_ylabel('mean centerline crossings', fontsize=9)
        _headroom(ax); _annot(ax, b, "{:.2f}")

        ax = axes[1, 2]; ax.axis('off')
        bc_txt = f"{BC_high:.2f}" if pd.notna(BC_high) else "n/a"
        ax.text(0.0, 0.98,
                "MODEL ADJUDICATION (Sec. 2.2 & 6)\n"
                "  DDM (Model 1):  AUC_high <= AUC_low\n"
                "  LCA (Model 2):  AUC_high  > AUC_low\n"
                f"  ΔAUC = {dAUC:.1f} px  ->  {verdict}\n\n"
                f"high-conflict trials n = {int(auc_high_vals.notna().sum())}\n"
                f"low-conflict  trials n = {int(auc_low_vals.notna().sum())}\n"
                f"bimodality (high AUC)  = {bc_txt}\n"
                "  (>0.555 -> mixture, not one shape)",
                va='top', ha='left', fontsize=9, family='monospace', transform=ax.transAxes)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(session_path, 'indices.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)




        try:
            age = df['age'].iloc[0] if 'age' in df.columns else "N/A"
            gender = df['gender'].iloc[0] if 'gender' in df.columns else "N/A"


            pd.DataFrame([{
                'session': session_name,
                'age': age,
                'gender': gender,
                'AUC_high': auc_high,
                'AUC_low': auc_low,
                'dAUC_high_minus_low': dAUC,
                'DCI': DCI,
                'BC_high': BC_high,
                'n_high': int(auc_high_vals.notna().sum()),
                'n_low': int(auc_low_vals.notna().sum()),
                'RSI': RSI,
                'FLIP_exploratory': FLIP
            }]).to_csv(os.path.join(session_path, 'indices.csv'), index=False)


            with open(os.path.join(session_path, 'scientific_summary.txt'), 'w', encoding='utf-8') as f:
                f.write("═══════════════════════════════════════════════════════\n")
                f.write("DECISION DYNAMICS PROFILE\n")
                f.write("═══════════════════════════════════════════════════════\n\n")
                f.write(f"Session: {session_name}\n")
                f.write("───────────────────────────────────────────────────────\n")
                f.write("TRAJECTORY CONFLICT (signed AUC; toward competitor = +)\n")
                f.write("───────────────────────────────────────────────────────\n")
                f.write(f"AUC high-conflict (Pos-Neg):      {auc_high:.3f}\n")
                f.write(f"AUC low-conflict  (Neutral-X):    {auc_low:.3f}\n")
                f.write(f"ΔAUC (high - low)  [primary]:     {dAUC:.3f}\n")
                f.write(f"DCI = ΔAUC / AUC_low [ratio]:     {DCI:.3f}\n")
                f.write(f"Bimodality coeff. (high AUC):     {BC_high:.3f} "
                        "(>0.555 → mixture, not one shape)\n\n")
                f.write("───────────────────────────────────────────────────────\n")
                f.write("MODEL ADJUDICATION (manuscript Sec. 2.2 & 6)\n")
                f.write("───────────────────────────────────────────────────────\n")
                f.write("The two generative models make opposite, falsifiable\n")
                f.write("predictions about whether high-conflict (Pos-Neg) pairs\n")
                f.write("are more or less curved than low-conflict pairs:\n")
                f.write("  Model 1  minimal drift-diffusion         -> AUC_high <= AUC_low\n")
                f.write("  Model 2  LCA + salience-weighted compet.  -> AUC_high  > AUC_low\n")
                f.write("Adjudication uses the robust difference contrast ΔAUC;\n")
                f.write("the ratio DCI flips sign when AUC_low < 0 and is fragile\n")
                f.write("(manuscript Sec. 4.2).\n")
                if pd.notna(dAUC):
                    verdict = "Model 2 (LCA)" if dAUC > 0 else "Model 1 (minimal DDM)"
                    f.write(f"Observed ΔAUC = {dAUC:+.3f}  ->  consistent with {verdict}\n\n")
                else:
                    f.write("Observed ΔAUC = n/a (insufficient main-block AUC)\n\n")
                f.write("───────────────────────────────────────────────────────\n")
                f.write("STABILITY\n")
                f.write("───────────────────────────────────────────────────────\n")
                f.write(f"RSI (maintained / valid SWAP):    {RSI:.3f}  (chance = 0.5)\n\n")
                f.write("───────────────────────────────────────────────────────\n")
                f.write("EXPLORATORY (excluded from primary framework)\n")
                f.write("───────────────────────────────────────────────────────\n")
                f.write(f"FLIP (mean centerline crossings): {FLIP:.3f}\n\n")
                f.write("───────────────────────────────────────────────────────\n")
                f.write("NOTE\n")
                f.write("───────────────────────────────────────────────────────\n")
                f.write("Indices are empirical projections of latent decision\n")
                f.write("dynamics, not direct measures of psychological constructs.\n")
                f.write("AUC is time-normalized and sign-retaining (Sec. 4.2).\n")
                f.write("No clinical interpretation is assumed.\n")
                f.write("\n═══════════════════════════════════════════════════════\n")

            print(f"[OK] Indices successfully calculated for {session_name}")
            print(f"  DCI={DCI:.3f} | dAUC={dAUC:.3f} | BC_high={BC_high:.3f} | "
                  f"RSI={RSI:.3f} | FLIP={FLIP:.3f}")

        except Exception as e:
            print(f"[ERROR] calculating indices for {session_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        plt.close('all')


if __name__ == "__main__":
    run_advanced_analysis()