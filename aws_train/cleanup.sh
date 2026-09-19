#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
source "$ROOT/resources.env"
echo "Cleaning Suraksha resources in $REGION ..."
if [[ -n "${INSTANCE_ID:-}" ]]; then
  aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" || true
  aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$INSTANCE_ID" || true
fi
if [[ -n "${SG_ID:-}" ]]; then
  aws ec2 delete-security-group --region "$REGION" --group-id "$SG_ID" || true
fi
if [[ -n "${KEY_NAME:-}" ]]; then
  aws ec2 delete-key-pair --region "$REGION" --key-name "$KEY_NAME" || true
fi
if [[ -n "${PROFILE_NAME:-}" && -n "${ROLE_NAME:-}" ]]; then
  aws iam remove-role-from-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME" || true
  aws iam delete-instance-profile --instance-profile-name "$PROFILE_NAME" || true
fi
if [[ -n "${ROLE_NAME:-}" ]]; then
  aws iam detach-role-policy --role-name "$ROLE_NAME" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore || true
  aws iam delete-role-policy --role-name "$ROLE_NAME" --policy-name SurakshaS3 || true
  aws iam delete-role --role-name "$ROLE_NAME" || true
fi
if [[ -n "${BUCKET:-}" ]]; then
  aws s3 rm "s3://${BUCKET}" --recursive || true
  aws s3api delete-bucket --bucket "$BUCKET" --region "$REGION" || true
fi
rm -f "$ROOT/suraksha.pem"
echo "Cleanup done."
