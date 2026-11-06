"""
보안 공지:
  이 스크립트는 더 이상 평문 자격증명으로 토큰을 발급하지 않습니다.
  PRAW(praw.ini) 기반 클라이언트를 사용해 인증을 확인하세요.

사용법:
  1) `configs/praw.ini.example`를 프로젝트 루트로 복사하여 `praw.ini`로 이름 변경
  2) Reddit 앱/계정 정보를 채운 뒤 저장(커밋 금지)
  3) 아래 스크립트를 실행해 인증이 정상 동작하는지 확인
"""

import sys

try:
    import praw  # type: ignore
except Exception as exc:  # pragma: no cover
    print("praw 패키지가 필요합니다. pip/conda로 설치 후 다시 시도하세요.")
    raise


def main() -> None:
    """PRAW 설정을 검사하고, 현재 사용자 정보를 출력합니다."""
    reddit = praw.Reddit("DEFAULT")  # 루트의 praw.ini [DEFAULT] 섹션 사용
    me = reddit.user.me()
    print("Authenticated as:", me)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # pragma: no cover
        print("인증 확인 실패:", e, file=sys.stderr)
        sys.exit(1)