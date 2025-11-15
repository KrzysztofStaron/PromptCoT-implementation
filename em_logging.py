# em_logging.py
# Logging helpers for EM training loop
import wandb
import logging

log = logging.getLogger(__name__)

# === WANDB LOGGING HELPERS ===

def log_batch_metrics(em_iter, batch_num, total_batches, batch_rewards, batch_selected_rewards, 
                      avg_rationale_length, batch_tiebreaker_used, batch_reward_spreads_eligible, 
                      batch_eligible_count, batch_size):
    """Log batch-level metrics to wandb"""
    avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
    max_reward = max(batch_rewards) if batch_rewards else 0
    min_reward = min(batch_rewards) if batch_rewards else 0
    std_reward = (sum((r - avg_reward) ** 2 for r in batch_rewards) / len(batch_rewards)) ** 0.5 if len(batch_rewards) > 1 else 0.0
    avg_selected_reward = sum(batch_selected_rewards) / len(batch_selected_rewards) if batch_selected_rewards else 0
    
    reward_spread_avg = sum(batch_reward_spreads_eligible) / len(batch_reward_spreads_eligible) if batch_reward_spreads_eligible else 0.0
    reward_spread_min = min(batch_reward_spreads_eligible) if batch_reward_spreads_eligible else 0.0
    reward_spread_max = max(batch_reward_spreads_eligible) if batch_reward_spreads_eligible else 0.0
    
    global_step = ((em_iter - 1) * total_batches) + batch_num
    wandb.log({
        "batch/iteration": em_iter,
        "batch/batch_num": batch_num,
        "batch/reward_avg_all": avg_reward,
        "batch/reward_avg_selected": avg_selected_reward,
        "batch/reward_max": max_reward,
        "batch/reward_min": min_reward,
        "batch/reward_std": std_reward,
        "batch/avg_rationale_length": avg_rationale_length,
        "batch/tiebreaker_used": batch_tiebreaker_used,
        "batch/reward_spread_avg": reward_spread_avg,
        "batch/reward_spread_min": reward_spread_min,
        "batch/reward_spread_max": reward_spread_max,
        "batch/reward_spread_eligible_count": batch_eligible_count,
    }, step=global_step)
    
    log.info(f"[E-STEP] Batch {batch_num} complete. Avg reward: {avg_reward:.2f}, Best: {max_reward:.2f}")
    log.info(f"[E-STEP] Batch {batch_num} summary: {batch_tiebreaker_used}/{batch_size} actually used tiebreaker ({batch_tiebreaker_used/batch_size*100:.1f}%), {batch_eligible_count} eligible")

def log_final_batch_metrics(em_iter, batch_num, total_batches, batch_rewards, batch_selected_rewards,
                            avg_rationale_length, batch_tiebreaker_used, batch_reward_spreads_eligible,
                            batch_eligible_count, batch_size):
    """Log final batch metrics to wandb"""
    avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
    max_reward = max(batch_rewards) if batch_rewards else 0
    min_reward = min(batch_rewards) if batch_rewards else 0
    std_reward = (sum((r - avg_reward) ** 2 for r in batch_rewards) / len(batch_rewards)) ** 0.5 if len(batch_rewards) > 1 else 0.0
    avg_selected_reward = sum(batch_selected_rewards) / len(batch_selected_rewards) if batch_selected_rewards else 0
    
    reward_spread_avg = sum(batch_reward_spreads_eligible) / len(batch_reward_spreads_eligible) if batch_reward_spreads_eligible else 0.0
    reward_spread_min = min(batch_reward_spreads_eligible) if batch_reward_spreads_eligible else 0.0
    reward_spread_max = max(batch_reward_spreads_eligible) if batch_reward_spreads_eligible else 0.0
    
    global_step = (em_iter * total_batches) + batch_num
    wandb.log({
        "batch/iteration": em_iter + 1,
        "batch/batch_num": batch_num,
        "batch/reward_avg_all": avg_reward,
        "batch/reward_avg_selected": avg_selected_reward,
        "batch/reward_max": max_reward,
        "batch/reward_min": min_reward,
        "batch/reward_std": std_reward,
        "batch/avg_rationale_length": avg_rationale_length,
        "batch/tiebreaker_used": batch_tiebreaker_used,
        "batch/reward_spread_avg": reward_spread_avg,
        "batch/reward_spread_min": reward_spread_min,
        "batch/reward_spread_max": reward_spread_max,
        "batch/reward_spread_eligible_count": batch_eligible_count,
    }, step=global_step)
    
    log.info(f"[E-STEP] Final batch complete. Avg reward: {avg_reward:.2f}, Best: {max_reward:.2f}")
    log.info(f"[E-STEP] Final batch summary: {batch_tiebreaker_used}/{batch_size} actually used tiebreaker ({batch_tiebreaker_used/batch_size*100:.1f}%), {batch_eligible_count} eligible")

def log_e_step_summary(em_iter, total_batches, all_rewards, total_tiebreaker_used):
    """Log E-step summary to wandb"""
    if all_rewards:
        avg_reward = sum(all_rewards) / len(all_rewards)
        max_reward = max(all_rewards)
        min_reward = min(all_rewards)
        std_reward = (sum((r - avg_reward) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5 if len(all_rewards) > 1 else 0.0
        log.info(f"[E-STEP] Complete! Selected {len(all_rewards)} triples")
        log.info(f"[E-STEP] Reward stats - Avg: {avg_reward:.2f}, Max: {max_reward:.2f}, Min: {min_reward:.2f}, Std: {std_reward:.2f}")
    else:
        avg_reward = 0.0
        max_reward = 0.0
        min_reward = 0.0
        std_reward = 0.0
    
    e_step_global_step = ((em_iter - 1) * total_batches) + total_batches
    wandb.log({
        "e_step/iteration": em_iter,
        "e_step/reward_avg": avg_reward,
        "e_step/reward_max": max_reward,
        "e_step/reward_min": min_reward,
        "e_step/reward_std": std_reward,
        "e_step/tiebreaker_used_total": total_tiebreaker_used,
    }, step=e_step_global_step)
    log.info(f"[WANDB] Logged E-step summary at step {em_iter}: reward_avg={avg_reward:.2f}, reward_max={max_reward:.2f}")

def log_m_step_summary(em_iter, e_step_global_step, prompt_loss, rationale_loss, 
                       prompt_structure_accuracy, rationale_structure_accuracy):
    """Log M-step summary to wandb"""
    m_step_global_step = e_step_global_step + 1
    wandb.log({
        "m_step/iteration": em_iter,
        "m_step/prompt_loss": prompt_loss,
        "m_step/rationale_loss": rationale_loss,
        "m_step/combined_loss": prompt_loss + rationale_loss,
        "m_step/prompt_structure_accuracy": prompt_structure_accuracy,
        "m_step/rationale_structure_accuracy": rationale_structure_accuracy,
    }, step=m_step_global_step)
    log.info(f"[WANDB] Logged M-step at step {m_step_global_step}: prompt_loss={prompt_loss:.4f}, rationale_loss={rationale_loss:.4f}")
    log.info(f"[WANDB] Structure accuracy - prompt: {prompt_structure_accuracy:.2%}, rationale: {rationale_structure_accuracy:.2%}")

def log_iteration_summary(em_iter, m_step_global_step, num_triples):
    """Log iteration summary to wandb"""
    iter_summary_step = m_step_global_step + 1
    wandb.log({
        "iteration/num": em_iter,
    }, step=iter_summary_step)
    log.info(f"[WANDB] Logged iteration {em_iter} complete: {num_triples} triples")

# === VERBOSE LOGGING HELPERS ===

def log_winner_rationale(best_idx, num_candidates, selected_reward, reward_spread, 
                        problem, rewards, rationale, tiebreaker_used=False):
    """Log details about the selected winning rationale"""
    if tiebreaker_used:
        log.info(f"[WINNER] 🎯 Tiebreaker selected rationale {best_idx+1}/{num_candidates} (reward={selected_reward:.2f}, spread={reward_spread:.3f})")
        log.info(f"[WINNER] Problem: {problem[:100]}..." if len(problem) > 100 else f"[WINNER] Problem: {problem}")
        log.info(f"[WINNER] All candidate rewards: {[f'{r:.3f}' for r in rewards]}")
        log.info(f"[WINNER] Rationale: {rationale[:150]}..." if len(rationale) > 150 else f"[WINNER] Rationale: {rationale}")
    else:
        log.info(f"[WINNER] Selected rationale {best_idx+1}/{num_candidates} (reward={selected_reward:.2f}, spread={reward_spread:.3f})")
        log.info(f"[WINNER] Rationale: {rationale[:150]}..." if len(rationale) > 150 else f"[WINNER] Rationale: {rationale}")

def log_batch_start(batch_num, total_batches, batch_size, current_k_samples):
    """Log batch processing start"""
    log.info(f"[E-STEP] Processing batch {batch_num}/{total_batches} ({batch_size} samples)")
    log.info(f"[E-STEP] Generating {current_k_samples} rationale candidates per sample...")

def log_batch_generation_complete(num_candidates):
    """Log completion of rationale generation"""
    log.info(f"[E-STEP] Generated {num_candidates} total candidates")
    log.info(f"[E-STEP] Computing rewards and selecting best rationale...")

def log_e_step_progress(sample_idx, batch_size):
    """Log progress within E-step batch processing"""
    if (sample_idx + 1) % 4 == 0:
        log.info(f"[E-STEP]   Processed {sample_idx+1}/{batch_size} samples in batch")

