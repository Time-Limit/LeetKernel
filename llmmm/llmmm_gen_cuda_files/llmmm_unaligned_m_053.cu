#include "llmmm/llmmm.cuh"

namespace LLMMM {

class UnalignedM_53_MMInstantiatorWrapper: public MMInstantiatorWrapper {
  MMInstantiator<53> mm_instantiator;

public:
  UnalignedM_53_MMInstantiatorWrapper()
  {
    mm_instantiator.apply();
  }
} __UnalignedM_53_MMInstantiatorWrapper__;

}  // namespace LLMMM
